#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace model {

namespace fs = std::filesystem;

#ifndef HSCUP_VERSION
#define HSCUP_VERSION "2.3.0-clipped-electrodes-spatial"
#endif

constexpr const char* kProgramVersion = HSCUP_VERSION;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kBoxLength = 10000.0;
constexpr double kHalfBox = 5000.0;
constexpr double kRodLength = 5000.0;
constexpr double kRodRadius = 30.0;
constexpr double kSphereRadius = 200.0;
constexpr double kGap = 1.8;
constexpr double kRodRodCutoff = 2.0 * kRodRadius + kGap;
constexpr double kRodSphereCutoff = kRodRadius + kSphereRadius + kGap;
constexpr double kSphereSphereCutoff = 2.0 * kSphereRadius + kGap;
constexpr double kCostA = 0.0148440253;
constexpr double kCostB = 0.0016755161;
constexpr std::uint64_t kSeedStride = 0x9e3779b97f4a7c15ULL;

struct Vec3 { double x{}, y{}, z{}; };
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator-(const Vec3& a) { return {-a.x,-a.y,-a.z}; }
inline Vec3 operator*(const Vec3& a, double s) { return {a.x*s,a.y*s,a.z*s}; }
inline Vec3 operator/(const Vec3& a, double s) { return {a.x/s,a.y/s,a.z/s}; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline double norm2(const Vec3& a) { return dot(a,a); }
inline double norm(const Vec3& a) { return std::sqrt(norm2(a)); }
inline Vec3 cross(const Vec3& a,const Vec3& b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}

struct Segment { Vec3 p, q; int source_id{}; int fragment_id{}; };
struct SphereImage { Vec3 center; int source_id{}; int fragment_id{}; };
struct Edge { int first{}, second{}; };
struct Candidate { int rods{}, spheres{}; };
struct ProbabilityResult { int rods{}, spheres{}; std::uint64_t successes{}, trials{}; };
enum class FragmentConnectivity { Independent, SourceConnected };

class DisjointSet {
public:
    explicit DisjointSet(std::size_t n): parent_(n), rank_(n,0) { std::iota(parent_.begin(),parent_.end(),0); }
    int find(int x) {
        while (parent_[static_cast<std::size_t>(x)] != x) {
            parent_[static_cast<std::size_t>(x)] = parent_[static_cast<std::size_t>(parent_[static_cast<std::size_t>(x)])];
            x = parent_[static_cast<std::size_t>(x)];
        }
        return x;
    }
    void unite(int a,int b) {
        a=find(a); b=find(b); if(a==b) return;
        if(rank_[static_cast<std::size_t>(a)]<rank_[static_cast<std::size_t>(b)]) std::swap(a,b);
        parent_[static_cast<std::size_t>(b)] = a;
        if(rank_[static_cast<std::size_t>(a)]==rank_[static_cast<std::size_t>(b)]) ++rank_[static_cast<std::size_t>(a)];
    }
private:
    std::vector<int> parent_;
    std::vector<unsigned char> rank_;
};

inline double point_segment_distance2(const Vec3& point,const Segment& segment) {
    const Vec3 d=segment.q-segment.p;
    const double l2=norm2(d);
    if(l2<1e-24) return norm2(point-segment.p);
    double t=dot(point-segment.p,d)/l2;
    t=std::clamp(t,0.0,1.0);
    return norm2(point-(segment.p+d*t));
}

inline double segment_segment_distance2(const Segment& first,const Segment& second) {
    const Vec3 u=first.q-first.p;
    const Vec3 v=second.q-second.p;
    const Vec3 w=first.p-second.p;
    const double a=dot(u,u), b=dot(u,v), c=dot(v,v), d=dot(u,w), e=dot(v,w);
    const double D=a*c-b*b;
    constexpr double eps=1e-12;
    double sN,sD=D,tN,tD=D;
    if(D<eps){ sN=0.0; sD=1.0; tN=e; tD=c; }
    else {
        sN=b*e-c*d; tN=a*e-b*d;
        if(sN<0.0){ sN=0.0; tN=e; tD=c; }
        else if(sN>sD){ sN=sD; tN=e+b; tD=c; }
    }
    if(tN<0.0){
        tN=0.0;
        if(-d<0.0) sN=0.0;
        else if(-d>a) sN=sD;
        else { sN=-d; sD=a; }
    } else if(tN>tD){
        tN=tD;
        if((-d+b)<0.0) sN=0.0;
        else if((-d+b)>a) sN=sD;
        else { sN=(-d+b); sD=a; }
    }
    const double sc=(std::abs(sN)<eps?0.0:sN/sD);
    const double tc=(std::abs(tN)<eps?0.0:tN/tD);
    return norm2(w+u*sc-v*tc);
}

Vec3 cylinder_support(const Segment& cylinder,const Vec3& direction) {
    const Vec3 axis_vector=cylinder.q-cylinder.p;
    const double length=norm(axis_vector);
    if(length<1e-12) throw std::invalid_argument("flat cylinder axis has zero length");
    const Vec3 axis=axis_vector/length;
    const Vec3 center=(cylinder.p+cylinder.q)*0.5;
    const double axial=dot(direction,axis);
    const Vec3 perpendicular=direction-axis*axial;
    Vec3 point=center+axis*((axial>=0.0?1.0:-1.0)*length*0.5);
    const double perpendicular_norm=norm(perpendicular);
    if(perpendicular_norm>1e-14) point=point+perpendicular*(kRodRadius/perpendicular_norm);
    return point;
}

Vec3 solid_cylinder_projection(const Vec3& point,const Segment& cylinder);

bool solid_cylinders_within_gap_projection(const Segment& first,const Segment& second,double gap) {
    const Vec3 second_center=(second.p+second.q)*0.5;
    Vec3 first_point=solid_cylinder_projection(second_center,first);
    Vec3 second_point=solid_cylinder_projection(first_point,second);
    double previous_distance=std::numeric_limits<double>::infinity();
    int stable_iterations=0;
    for(int iteration=0;iteration<10000;++iteration){
        const Vec3 next_first=solid_cylinder_projection(second_point,first);
        const Vec3 next_second=solid_cylinder_projection(next_first,second);
        const double distance=norm(next_second-next_first);
        if(distance<=gap+1e-8) return true;
        if(std::abs(previous_distance-distance)<=1e-9) ++stable_iterations;
        else stable_iterations=0;
        if(stable_iterations>=8) return false;
        first_point=next_first;
        second_point=next_second;
        previous_distance=distance;
    }
    return norm(second_point-first_point)<=gap+1e-7;
}

Vec3 minkowski_support_with_gap(const Segment& first,const Segment& second,
                                const Vec3& raw_direction,double gap) {
    Vec3 direction=raw_direction;
    double direction_norm=norm(direction);
    if(direction_norm<1e-14){direction={1.0,0.0,0.0};direction_norm=1.0;}
    return cylinder_support(first,direction)-cylinder_support(second,-direction)
           +direction*(gap/direction_norm);
}

Vec3 triple_cross(const Vec3& a,const Vec3& b,const Vec3& c) {
    return cross(cross(a,b),c);
}

bool same_direction(const Vec3& direction,const Vec3& toward) {
    return dot(direction,toward)>0.0;
}

struct GjkSimplex {
    std::array<Vec3,4> points{};
    std::size_t size{};
    void push_front(const Vec3& point) {
        for(std::size_t i=std::min<std::size_t>(size,3U);i>0U;--i) points[i]=points[i-1U];
        points[0]=point;
        size=std::min<std::size_t>(size+1U,4U);
    }
};

bool update_line_simplex(GjkSimplex& simplex,Vec3& direction) {
    const Vec3 a=simplex.points[0],b=simplex.points[1],ab=b-a,ao=-a;
    if(same_direction(ab,ao)){
        direction=triple_cross(ab,ao,ab);
        if(norm2(direction)<1e-24) return true;
    }else{
        simplex.points[0]=a;
        simplex.size=1U;
        direction=ao;
    }
    return norm2(direction)<1e-24;
}

bool update_triangle_simplex(GjkSimplex& simplex,Vec3& direction) {
    const Vec3 a=simplex.points[0],b=simplex.points[1],c=simplex.points[2];
    const Vec3 ab=b-a,ac=c-a,ao=-a,abc=cross(ab,ac);
    if(norm2(abc)<1e-24){
        simplex.points[1]=(norm2(ab)>=norm2(ac)?b:c);
        simplex.size=2U;
        return update_line_simplex(simplex,direction);
    }
    if(same_direction(cross(abc,ac),ao)){
        if(same_direction(ac,ao)){
            simplex.points[1]=c;
            simplex.size=2U;
            direction=triple_cross(ac,ao,ac);
            if(norm2(direction)<1e-24) return true;
        }else{
            simplex.points[1]=b;
            simplex.size=2U;
            return update_line_simplex(simplex,direction);
        }
    }else if(same_direction(cross(ab,abc),ao)){
        simplex.points[1]=b;
        simplex.size=2U;
        return update_line_simplex(simplex,direction);
    }else if(same_direction(abc,ao)){
        direction=abc;
    }else{
        simplex.points[1]=c;
        simplex.points[2]=b;
        direction=-abc;
    }
    return norm2(direction)<1e-24;
}

bool update_tetrahedron_simplex(GjkSimplex& simplex,Vec3& direction) {
    const Vec3 a=simplex.points[0],b=simplex.points[1],c=simplex.points[2],d=simplex.points[3],ao=-a;
    const Vec3 abc=cross(b-a,c-a),acd=cross(c-a,d-a),adb=cross(d-a,b-a);
    if(same_direction(abc,ao)){
        simplex.points[1]=b;simplex.points[2]=c;simplex.size=3U;
        return update_triangle_simplex(simplex,direction);
    }
    if(same_direction(acd,ao)){
        simplex.points[1]=c;simplex.points[2]=d;simplex.size=3U;
        return update_triangle_simplex(simplex,direction);
    }
    if(same_direction(adb,ao)){
        simplex.points[1]=d;simplex.points[2]=b;simplex.size=3U;
        return update_triangle_simplex(simplex,direction);
    }
    return true;
}

bool update_gjk_simplex(GjkSimplex& simplex,Vec3& direction) {
    if(simplex.size==2U) return update_line_simplex(simplex,direction);
    if(simplex.size==3U) return update_triangle_simplex(simplex,direction);
    if(simplex.size==4U) return update_tetrahedron_simplex(simplex,direction);
    direction=-simplex.points[0];
    return norm2(direction)<1e-24;
}

bool flat_cylinders_within_gap(const Segment& first,const Segment& second,double gap) {
    const double capsule_cutoff=2.0*kRodRadius+gap;
    if(segment_segment_distance2(first,second)>capsule_cutoff*capsule_cutoff) return false;
    const Vec3 first_center=(first.p+first.q)*0.5,second_center=(second.p+second.q)*0.5;
    Vec3 direction=second_center-first_center;
    if(norm2(direction)<1e-24) direction={1.0,0.0,0.0};
    GjkSimplex simplex;
    simplex.push_front(minkowski_support_with_gap(first,second,direction,gap+1e-9));
    direction=-simplex.points[0];
    for(int iteration=0;iteration<64;++iteration){
        if(norm2(direction)<1e-22) return true;
        const Vec3 support=minkowski_support_with_gap(first,second,direction,gap+1e-9);
        if(dot(support,direction)<0.0) return false;
        bool duplicate=false;
        for(std::size_t i=0;i<simplex.size;++i){
            if(norm2(support-simplex.points[i])<1e-20){duplicate=true;break;}
        }
        if(duplicate) return solid_cylinders_within_gap_projection(first,second,gap);
        simplex.push_front(support);
        if(update_gjk_simplex(simplex,direction)) return true;
    }
    return solid_cylinders_within_gap_projection(first,second,gap);
}

double flat_cylinder_surface_gap(const Segment& first,const Segment& second,double upper_bound) {
    if(flat_cylinders_within_gap(first,second,0.0)) return 0.0;
    if(!flat_cylinders_within_gap(first,second,upper_bound)) {
        throw std::invalid_argument("cylinder gap exceeds supplied upper bound");
    }
    double low=0.0,high=upper_bound;
    for(int iteration=0;iteration<52;++iteration){
        const double mid=(low+high)*0.5;
        if(flat_cylinders_within_gap(first,second,mid)) high=mid;
        else low=mid;
    }
    return high;
}

double point_flat_cylinder_distance(const Vec3& point,const Segment& cylinder) {
    const Vec3 axis_vector=cylinder.q-cylinder.p;
    const double length=norm(axis_vector);
    if(length<1e-12) throw std::invalid_argument("flat cylinder axis has zero length");
    const Vec3 axis=axis_vector/length;
    const Vec3 center=(cylinder.p+cylinder.q)*0.5;
    const Vec3 relative=point-center;
    const double axial=dot(relative,axis);
    const double radial=std::sqrt(std::max(0.0,norm2(relative)-axial*axial));
    const double radial_excess=std::max(0.0,radial-kRodRadius);
    const double axial_excess=std::max(0.0,std::abs(axial)-length*0.5);
    return std::hypot(radial_excess,axial_excess);
}

bool flat_cylinder_sphere_within_gap(const Segment& cylinder,const Vec3& sphere_center,double gap) {
    return point_flat_cylinder_distance(sphere_center,cylinder)<=kSphereRadius+gap+1e-9;
}

double coordinate(const Vec3& point,int axis) {
    if(axis==0) return point.x;
    if(axis==1) return point.y;
    if(axis==2) return point.z;
    throw std::invalid_argument("axis index must be 0, 1, or 2");
}

std::pair<double,double> flat_cylinder_axis_range(const Segment& cylinder,int axis) {
    const Vec3 axis_vector=cylinder.q-cylinder.p;
    const double length=norm(axis_vector);
    if(length<1e-12) throw std::invalid_argument("flat cylinder axis has zero length");
    const double component=coordinate(axis_vector,axis)/length;
    const double center=(coordinate(cylinder.p,axis)+coordinate(cylinder.q,axis))*0.5;
    const double extent=length*0.5*std::abs(component)
                        +kRodRadius*std::sqrt(std::max(0.0,1.0-component*component));
    return {center-extent,center+extent};
}

std::pair<double,double> flat_cylinder_x_range(const Segment& cylinder) {
    return flat_cylinder_axis_range(cylinder,0);
}

Vec3 aabb_projection(const Vec3& point,const Vec3& lower,const Vec3& upper) {
    return {
        std::clamp(point.x,lower.x,upper.x),
        std::clamp(point.y,lower.y,upper.y),
        std::clamp(point.z,lower.z,upper.z)
    };
}

Vec3 aabb_support(const Vec3& direction,const Vec3& lower,const Vec3& upper) {
    return {
        direction.x>=0.0?upper.x:lower.x,
        direction.y>=0.0?upper.y:lower.y,
        direction.z>=0.0?upper.z:lower.z
    };
}

Vec3 box_projection(const Vec3& point) {
    return aabb_projection(point,{-kHalfBox,-kHalfBox,-kHalfBox},
                           { kHalfBox, kHalfBox, kHalfBox});
}

Vec3 box_support(const Vec3& direction) {
    return aabb_support(direction,{-kHalfBox,-kHalfBox,-kHalfBox},
                        { kHalfBox, kHalfBox, kHalfBox});
}

Vec3 solid_cylinder_projection(const Vec3& point,const Segment& cylinder) {
    const Vec3 axis_vector=cylinder.q-cylinder.p;
    const double length=norm(axis_vector);
    if(length<1e-12) throw std::invalid_argument("flat cylinder axis has zero length");
    const Vec3 axis=axis_vector/length;
    const Vec3 center=(cylinder.p+cylinder.q)*0.5;
    const Vec3 relative=point-center;
    const double raw_axial=dot(relative,axis);
    const double axial=std::clamp(raw_axial,-length*0.5,length*0.5);
    const Vec3 radial=relative-axis*raw_axial;
    const double radial_norm=norm(radial);
    const Vec3 bounded_radial=radial_norm>kRodRadius
        ?radial*(kRodRadius/radial_norm):radial;
    return center+axis*axial+bounded_radial;
}

Vec3 solid_sphere_projection(const Vec3& point,const Vec3& center) {
    const Vec3 offset=point-center;
    const double distance=norm(offset);
    if(distance<=kSphereRadius) return point;
    if(distance<1e-14) return center;
    return center+offset*(kSphereRadius/distance);
}

struct LinearConstraint {
    Vec3 normal;
    double bound{};
};

int selected_constraint_count(unsigned mask) {
    int count=0;
    while(mask!=0U){count+=static_cast<int>(mask&1U);mask>>=1U;}
    return count;
}

bool solve_small_linear_system(std::array<std::array<double,3>,3> matrix,
                               std::array<double,3> rhs,int size,
                               std::array<double,3>& solution) {
    solution={0.0,0.0,0.0};
    for(int column=0;column<size;++column){
        int pivot=column;
        for(int row=column+1;row<size;++row){
            if(std::abs(matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)])>
               std::abs(matrix[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(column)]))
                pivot=row;
        }
        if(std::abs(matrix[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(column)])<1e-12)
            return false;
        if(pivot!=column){
            std::swap(matrix[static_cast<std::size_t>(pivot)],matrix[static_cast<std::size_t>(column)]);
            std::swap(rhs[static_cast<std::size_t>(pivot)],rhs[static_cast<std::size_t>(column)]);
        }
        const double divisor=matrix[static_cast<std::size_t>(column)][static_cast<std::size_t>(column)];
        for(int entry=column;entry<size;++entry)
            matrix[static_cast<std::size_t>(column)][static_cast<std::size_t>(entry)]/=divisor;
        rhs[static_cast<std::size_t>(column)]/=divisor;
        for(int row=0;row<size;++row){
            if(row==column) continue;
            const double factor=matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)];
            for(int entry=column;entry<size;++entry)
                matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(entry)]-=
                    factor*matrix[static_cast<std::size_t>(column)][static_cast<std::size_t>(entry)];
            rhs[static_cast<std::size_t>(row)]-=factor*rhs[static_cast<std::size_t>(column)];
        }
    }
    for(int row=0;row<size;++row) solution[static_cast<std::size_t>(row)]=rhs[static_cast<std::size_t>(row)];
    return true;
}

Vec3 cylinder_hessian_inverse(const Vec3& vector,const Vec3& axis,double multiplier) {
    const Vec3 axial=axis*dot(vector,axis);
    return axial+(vector-axial)/(1.0+multiplier);
}

Vec3 active_set_cylinder_aabb_projection(const Vec3& point,const Segment& cylinder,
                                         const Vec3& lower,const Vec3& upper) {
    const Vec3 axis_vector=cylinder.q-cylinder.p;
    const double length=norm(axis_vector);
    if(length<1e-12) throw std::invalid_argument("flat cylinder axis has zero length");
    const Vec3 axis=axis_vector/length;
    const Vec3 center=(cylinder.p+cylinder.q)*0.5;
    const Vec3 target=point-center;
    const double half_length=length*0.5;
    const std::array<LinearConstraint,8> constraints{{
        {{ 1.0, 0.0, 0.0}, upper.x-center.x},
        {{-1.0, 0.0, 0.0}, center.x-lower.x},
        {{ 0.0, 1.0, 0.0}, upper.y-center.y},
        {{ 0.0,-1.0, 0.0}, center.y-lower.y},
        {{ 0.0, 0.0, 1.0}, upper.z-center.z},
        {{ 0.0, 0.0,-1.0}, center.z-lower.z},
        {axis,half_length},
        {-axis,half_length}
    }};
    double best_objective=std::numeric_limits<double>::infinity();
    Vec3 best{};
    bool found=false;
    constexpr double feasibility_tolerance=2e-7;
    for(unsigned mask=0U;mask<(1U<<constraints.size());++mask){
        const int active_count=selected_constraint_count(mask);
        if(active_count>3) continue;
        std::array<int,3> active{};
        int active_cursor=0;
        for(std::size_t index=0;index<constraints.size();++index){
            if((mask&(1U<<index))!=0U) active[static_cast<std::size_t>(active_cursor++)]=static_cast<int>(index);
        }
        auto solve_at_multiplier=[&](double multiplier,Vec3& relative,
                                     std::array<double,3>& multipliers){
            const Vec3 inverse_target=cylinder_hessian_inverse(target,axis,multiplier);
            std::array<std::array<double,3>,3> matrix{};
            std::array<double,3> rhs{};
            std::array<Vec3,3> inverse_normals{};
            for(int row=0;row<active_count;++row){
                const auto& row_constraint=constraints[static_cast<std::size_t>(active[static_cast<std::size_t>(row)])];
                inverse_normals[static_cast<std::size_t>(row)]=
                    cylinder_hessian_inverse(row_constraint.normal,axis,multiplier);
                rhs[static_cast<std::size_t>(row)]=
                    dot(row_constraint.normal,inverse_target)-row_constraint.bound;
                for(int column=0;column<active_count;++column){
                    const auto& column_constraint=constraints[static_cast<std::size_t>(active[static_cast<std::size_t>(column)])];
                    matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)]=
                        dot(row_constraint.normal,
                            cylinder_hessian_inverse(column_constraint.normal,axis,multiplier));
                }
            }
            if(!solve_small_linear_system(matrix,rhs,active_count,multipliers)) return false;
            relative=inverse_target;
            for(int index=0;index<active_count;++index)
                relative=relative-inverse_normals[static_cast<std::size_t>(index)]*
                    multipliers[static_cast<std::size_t>(index)];
            return true;
        };
        auto accept_candidate=[&](const Vec3& relative,const std::array<double,3>& multipliers,
                                  bool radial_active){
            for(int index=0;index<active_count;++index){
                if(multipliers[static_cast<std::size_t>(index)]<-feasibility_tolerance) return;
            }
            for(const auto& constraint:constraints){
                if(dot(constraint.normal,relative)>constraint.bound+feasibility_tolerance) return;
            }
            const Vec3 radial=relative-axis*dot(relative,axis);
            const double radial2=norm2(radial);
            if(radial2>kRodRadius*kRodRadius+feasibility_tolerance) return;
            if(radial_active &&
               std::abs(radial2-kRodRadius*kRodRadius)>2e-4) return;
            const double objective=norm2(relative-target);
            if(objective<best_objective){
                best_objective=objective;
                best=relative;
                found=true;
            }
        };
        Vec3 inactive_relative{};
        std::array<double,3> inactive_multipliers{};
        if(!solve_at_multiplier(0.0,inactive_relative,inactive_multipliers)) continue;
        const Vec3 inactive_radial=inactive_relative-axis*dot(inactive_relative,axis);
        if(norm2(inactive_radial)<=kRodRadius*kRodRadius+feasibility_tolerance)
            accept_candidate(inactive_relative,inactive_multipliers,false);
        else{
            double low=0.0,high=1.0;
            Vec3 high_relative{};
            std::array<double,3> high_multipliers{};
            bool bracketed=false;
            for(int expansion=0;expansion<80;++expansion){
                if(!solve_at_multiplier(high,high_relative,high_multipliers)) break;
                const Vec3 high_radial=high_relative-axis*dot(high_relative,axis);
                if(norm2(high_radial)<=kRodRadius*kRodRadius){bracketed=true;break;}
                high*=2.0;
            }
            if(!bracketed) continue;
            for(int iteration=0;iteration<80;++iteration){
                const double middle=(low+high)*0.5;
                Vec3 middle_relative{};
                std::array<double,3> middle_multipliers{};
                if(!solve_at_multiplier(middle,middle_relative,middle_multipliers)){
                    low=middle;
                    continue;
                }
                const Vec3 middle_radial=middle_relative-axis*dot(middle_relative,axis);
                if(norm2(middle_radial)>kRodRadius*kRodRadius) low=middle;
                else high=middle;
            }
            if(solve_at_multiplier(high,high_relative,high_multipliers))
                accept_candidate(high_relative,high_multipliers,true);
        }
    }
    if(!found){
        std::ostringstream message;
        message<<"active-set cylinder-box projection found no feasible point: point=("
               <<point.x<<','<<point.y<<','<<point.z<<"), cylinder_p=("
               <<cylinder.p.x<<','<<cylinder.p.y<<','<<cylinder.p.z<<"), cylinder_q=("
               <<cylinder.q.x<<','<<cylinder.q.y<<','<<cylinder.q.z<<')';
        throw std::runtime_error(message.str());
    }
    return center+best;
}

Vec3 active_set_clipped_cylinder_projection(const Vec3& point,const Segment& cylinder) {
    return active_set_cylinder_aabb_projection(
        point,cylinder,{-kHalfBox,-kHalfBox,-kHalfBox},{kHalfBox,kHalfBox,kHalfBox});
}

Vec3 clipped_cylinder_projection(const Vec3& point,const Segment& cylinder) {
    Vec3 current=point;
    Vec3 primitive_correction{},box_correction{};
    for(int iteration=0;iteration<256;++iteration){
        const Vec3 primitive_input=current+primitive_correction;
        const Vec3 primitive_point=solid_cylinder_projection(primitive_input,cylinder);
        primitive_correction=primitive_input-primitive_point;
        const Vec3 box_input=primitive_point+box_correction;
        const Vec3 next=box_projection(box_input);
        box_correction=box_input-next;
        const double movement2=norm2(next-current);
        current=next;
        const double residual2=norm2(solid_cylinder_projection(current,cylinder)-current);
        if(movement2<=1e-16 && residual2<=1e-12) return current;
    }
    return active_set_clipped_cylinder_projection(point,cylinder);
}

Vec3 clipped_sphere_projection(const Vec3& point,const Vec3& center) {
    const Vec3 box_point=box_projection(point);
    if(norm2(box_point-center)<=kSphereRadius*kSphereRadius) return box_point;
    const Vec3 sphere_point=solid_sphere_projection(point,center);
    if(norm2(box_projection(sphere_point)-sphere_point)<=1e-20) return sphere_point;
    auto lagrange_point=[&](double multiplier){
        return box_projection((point+center*multiplier)/(1.0+multiplier));
    };
    double low=0.0,high=1.0;
    while(norm2(lagrange_point(high)-center)>kSphereRadius*kSphereRadius){
        high*=2.0;
        if(high>1e16) throw std::runtime_error("sphere-box intersection is empty");
    }
    for(int iteration=0;iteration<96;++iteration){
        const double middle=(low+high)*0.5;
        if(norm2(lagrange_point(middle)-center)>kSphereRadius*kSphereRadius) low=middle;
        else high=middle;
    }
    return lagrange_point(high);
}

template<class FirstProjection,class SecondProjection>
bool clipped_shapes_within_gap(const Vec3& first_seed,const Vec3& second_seed,double gap,
                               FirstProjection project_first,SecondProjection project_second) {
    Vec3 first=project_first(second_seed);
    Vec3 second=project_second(first_seed);
    double previous_distance=std::numeric_limits<double>::infinity();
    int stable_distance_iterations=0;
    for(int iteration=0;iteration<2048;++iteration){
        const Vec3 midpoint=(first+second)*0.5;
        const Vec3 next_first=project_first(midpoint);
        const Vec3 next_second=project_second(midpoint);
        const double distance=norm(next_second-next_first);
        if(distance<=gap+1e-8) return true;
        first=next_first;
        second=next_second;
        if(std::abs(previous_distance-distance)<=1e-8) ++stable_distance_iterations;
        else stable_distance_iterations=0;
        if(stable_distance_iterations>=8) return false;
        previous_distance=distance;
    }
    if(std::abs(previous_distance-norm(second-first))<=1e-6) return false;
    std::ostringstream message;
    message<<"clipped-fragment distance failed to converge: first_seed=("
           <<first_seed.x<<','<<first_seed.y<<','<<first_seed.z<<"), second_seed=("
           <<second_seed.x<<','<<second_seed.y<<','<<second_seed.z<<"), distance="
           <<norm(second-first)<<", previous_distance="<<previous_distance;
    throw std::runtime_error(message.str());
}

bool cylinder_inside_box(const Segment& cylinder) {
    for(int axis=0;axis<3;++axis){
        const auto range=flat_cylinder_axis_range(cylinder,axis);
        if(range.first<-kHalfBox || range.second>kHalfBox) return false;
    }
    return true;
}

bool sphere_inside_box(const Vec3& center) {
    return center.x-kSphereRadius>=-kHalfBox && center.x+kSphereRadius<=kHalfBox &&
           center.y-kSphereRadius>=-kHalfBox && center.y+kSphereRadius<=kHalfBox &&
           center.z-kSphereRadius>=-kHalfBox && center.z+kSphereRadius<=kHalfBox;
}

bool sphere_intersects_aabb(const Vec3& center,const Vec3& lower,const Vec3& upper) {
    return norm2(center-aabb_projection(center,lower,upper))<=
           kSphereRadius*kSphereRadius+1e-9;
}

bool sphere_intersects_box(const Vec3& center) {
    return sphere_intersects_aabb(
        center,{-kHalfBox,-kHalfBox,-kHalfBox},{kHalfBox,kHalfBox,kHalfBox});
}

bool flat_cylinder_intersects_aabb(const Segment& cylinder,
                                   const Vec3& lower,const Vec3& upper) {
    auto exact_fallback=[&](){
        try{
            static_cast<void>(active_set_cylinder_aabb_projection(
                aabb_projection((cylinder.p+cylinder.q)*0.5,lower,upper),
                cylinder,lower,upper));
            return true;
        }catch(const std::runtime_error&){
            return false;
        }
    };
    Vec3 direction=-(cylinder.p+cylinder.q)*0.5;
    if(norm2(direction)<1e-24) direction={1.0,0.0,0.0};
    auto support=[&](const Vec3& raw_direction){
        return cylinder_support(cylinder,raw_direction)-
               aabb_support(-raw_direction,lower,upper);
    };
    GjkSimplex simplex;
    simplex.push_front(support(direction));
    direction=-simplex.points[0];
    for(int iteration=0;iteration<64;++iteration){
        if(norm2(direction)<1e-20) return true;
        const Vec3 point=support(direction);
        const double separating_projection=dot(point,direction);
        if(separating_projection<-1e-8) return false;
        if(separating_projection<0.0) return exact_fallback();
        bool duplicate=false;
        for(std::size_t i=0;i<simplex.size;++i){
            if(norm2(point-simplex.points[i])<1e-18){duplicate=true;break;}
        }
        if(duplicate) return exact_fallback();
        simplex.push_front(point);
        if(update_gjk_simplex(simplex,direction)) return true;
    }
    return exact_fallback();
}

bool flat_cylinder_intersects_box(const Segment& cylinder) {
    return flat_cylinder_intersects_aabb(
        cylinder,{-kHalfBox,-kHalfBox,-kHalfBox},{kHalfBox,kHalfBox,kHalfBox});
}

bool fragment_flat_cylinders_within_gap(const Segment& first,const Segment& second,double gap) {
    if(!flat_cylinders_within_gap(first,second,gap)) return false;
    if(cylinder_inside_box(first) && cylinder_inside_box(second)) return true;
    return clipped_shapes_within_gap(
        (first.p+first.q)*0.5,(second.p+second.q)*0.5,gap,
        [&](const Vec3& point){return clipped_cylinder_projection(point,first);},
        [&](const Vec3& point){return clipped_cylinder_projection(point,second);});
}

bool fragment_flat_cylinder_sphere_within_gap(const Segment& cylinder,
                                               const Vec3& sphere_center,double gap) {
    if(!flat_cylinder_sphere_within_gap(cylinder,sphere_center,gap)) return false;
    if(cylinder_inside_box(cylinder) && sphere_inside_box(sphere_center)) return true;
    return clipped_shapes_within_gap(
        (cylinder.p+cylinder.q)*0.5,sphere_center,gap,
        [&](const Vec3& point){return clipped_cylinder_projection(point,cylinder);},
        [&](const Vec3& point){return clipped_sphere_projection(point,sphere_center);});
}

bool fragment_spheres_within_gap(const Vec3& first,const Vec3& second,double gap) {
    const double cutoff=2.0*kSphereRadius+gap;
    if(norm2(second-first)>cutoff*cutoff+1e-9) return false;
    if(sphere_inside_box(first) && sphere_inside_box(second)) return true;
    return clipped_shapes_within_gap(
        first,second,gap,
        [&](const Vec3& point){return clipped_sphere_projection(point,first);},
        [&](const Vec3& point){return clipped_sphere_projection(point,second);});
}

Segment translate_segment(const Segment& segment,const Vec3& shift) {
    return {segment.p+shift,segment.q+shift,segment.source_id,segment.fragment_id};
}

bool bbox_maybe_contact(const Segment& a,const Segment& b,double cutoff);

std::vector<Segment> wrapped_rod_images(const Segment& rod) {
    std::array<std::vector<double>,3> shifts;
    for(int axis=0;axis<3;++axis){
        shifts[static_cast<std::size_t>(axis)].push_back(0.0);
        const auto range=flat_cylinder_axis_range(rod,axis);
        if(range.first < -kHalfBox) shifts[static_cast<std::size_t>(axis)].push_back(kBoxLength);
        if(range.second > kHalfBox) shifts[static_cast<std::size_t>(axis)].push_back(-kBoxLength);
    }
    std::vector<Segment> out;
    int fragment_id=0;
    for(double dx:shifts[0]) for(double dy:shifts[1]) for(double dz:shifts[2]){
        Segment image=translate_segment(rod,{dx,dy,dz});
        const int shifted_axes=static_cast<int>(dx!=0.0)+static_cast<int>(dy!=0.0)+
                               static_cast<int>(dz!=0.0);
        if(shifted_axes>=1 && !flat_cylinder_intersects_box(image)) continue;
        image.fragment_id=fragment_id++;
        out.push_back(image);
    }
    return out;
}

std::vector<SphereImage> wrapped_sphere_images(const Vec3& center,int source_id);

bool periodic_flat_cylinders_within_gap(const Segment& first,const Segment& second,double gap) {
    const auto first_images=wrapped_rod_images(first);
    const auto second_images=wrapped_rod_images(second);
    for(const Segment& first_image:first_images) for(const Segment& second_image:second_images){
        if(!bbox_maybe_contact(first_image,second_image,2.0*kRodRadius+gap)) continue;
        if(fragment_flat_cylinders_within_gap(first_image,second_image,gap)) return true;
    }
    return false;
}

bool periodic_flat_cylinder_sphere_within_gap(const Segment& cylinder,
                                               const Vec3& sphere_center,double gap) {
    const auto cylinder_images=wrapped_rod_images(cylinder);
    const auto sphere_images=wrapped_sphere_images(sphere_center,0);
    for(const Segment& cylinder_image:cylinder_images) for(const SphereImage& sphere_image:sphere_images){
        if(fragment_flat_cylinder_sphere_within_gap(cylinder_image,sphere_image.center,gap)) return true;
    }
    return false;
}

bool periodic_spheres_within_gap(const Vec3& first,const Vec3& second,double gap) {
    const auto first_images=wrapped_sphere_images(first,0);
    const auto second_images=wrapped_sphere_images(second,1);
    for(const SphereImage& first_image:first_images) for(const SphereImage& second_image:second_images){
        if(fragment_spheres_within_gap(first_image.center,second_image.center,gap)) return true;
    }
    return false;
}

std::pair<unsigned char,unsigned char> rod_electrode_contacts(const Segment& rod) {
    const auto range=flat_cylinder_x_range(rod);
    const bool left=range.first<=-kHalfBox+kGap &&
        flat_cylinder_intersects_aabb(
            rod,{-kHalfBox,-kHalfBox,-kHalfBox},
            {-kHalfBox+kGap,kHalfBox,kHalfBox});
    const bool right=range.second>=kHalfBox-kGap &&
        flat_cylinder_intersects_aabb(
            rod,{kHalfBox-kGap,-kHalfBox,-kHalfBox},
            {kHalfBox,kHalfBox,kHalfBox});
    return {static_cast<unsigned char>(left),static_cast<unsigned char>(right)};
}

std::pair<unsigned char,unsigned char> sphere_electrode_contacts(const Vec3& center) {
    const bool left=center.x-kSphereRadius<=-kHalfBox+kGap &&
        sphere_intersects_aabb(
            center,{-kHalfBox,-kHalfBox,-kHalfBox},
            {-kHalfBox+kGap,kHalfBox,kHalfBox});
    const bool right=center.x+kSphereRadius>= kHalfBox-kGap &&
        sphere_intersects_aabb(
            center,{kHalfBox-kGap,-kHalfBox,-kHalfBox},
            {kHalfBox,kHalfBox,kHalfBox});
    return {static_cast<unsigned char>(left),static_cast<unsigned char>(right)};
}

std::vector<SphereImage> wrapped_sphere_images(const Vec3& center,int source_id) {
    std::vector<double> xs{0.0},ys{0.0},zs{0.0};
    if(center.x-kSphereRadius<-kHalfBox) xs.push_back(kBoxLength);
    if(center.x+kSphereRadius> kHalfBox) xs.push_back(-kBoxLength);
    if(center.y-kSphereRadius<-kHalfBox) ys.push_back(kBoxLength);
    if(center.y+kSphereRadius> kHalfBox) ys.push_back(-kBoxLength);
    if(center.z-kSphereRadius<-kHalfBox) zs.push_back(kBoxLength);
    if(center.z+kSphereRadius> kHalfBox) zs.push_back(-kBoxLength);
    std::vector<SphereImage> out;
    int fragment_id=0;
    for(double dx:xs) for(double dy:ys) for(double dz:zs){
        const Vec3 image{center.x+dx,center.y+dy,center.z+dz};
        if(!sphere_intersects_box(image)) continue;
        out.push_back({image,source_id,fragment_id++});
    }
    return out;
}

Vec3 random_isotropic_direction(std::mt19937_64& gen){
    std::uniform_real_distribution<double> u(0.0,1.0),phi(0.0,2.0*kPi);
    const double z=2.0*u(gen)-1.0, p=phi(gen), r=std::sqrt(std::max(0.0,1.0-z*z));
    return {r*std::cos(p),r*std::sin(p),z};
}
std::uint64_t trial_seed(std::uint64_t base,std::uint64_t i){ return base+kSeedStride*(i+1ULL); }
unsigned thread_count(unsigned req,std::uint64_t trials){
    unsigned n=req?req:std::thread::hardware_concurrency(); if(!n)n=1;
    return static_cast<unsigned>(std::min<std::uint64_t>(n,std::max<std::uint64_t>(1,trials)));
}

bool bbox_maybe_contact(const Segment& a,const Segment& b,double cutoff){
    return !(std::max(b.p.x,b.q.x)<std::min(a.p.x,a.q.x)-cutoff ||
             std::min(b.p.x,b.q.x)>std::max(a.p.x,a.q.x)+cutoff ||
             std::max(b.p.y,b.q.y)<std::min(a.p.y,a.q.y)-cutoff ||
             std::min(b.p.y,b.q.y)>std::max(a.p.y,a.q.y)+cutoff ||
             std::max(b.p.z,b.q.z)<std::min(a.p.z,a.q.z)-cutoff ||
             std::min(b.p.z,b.q.z)>std::max(a.p.z,a.q.z)+cutoff);
}

std::vector<ProbabilityResult> simulate_a_prefix(int min_rods,int max_rods,std::uint64_t trials,
                                                   std::uint64_t seed,unsigned requested_threads,
                                                   FragmentConnectivity connectivity=FragmentConnectivity::Independent){
    if(min_rods<0 || max_rods<min_rods || trials==0) throw std::invalid_argument("invalid A-prefix arguments");
    const std::size_t nres=static_cast<std::size_t>(max_rods-min_rods+1);
    const unsigned nt=thread_count(requested_threads,trials);
    std::vector<std::vector<std::uint64_t>> local(nt,std::vector<std::uint64_t>(nres,0));
    std::vector<std::thread> workers;
    for(unsigned tid=0;tid<nt;++tid){
        workers.emplace_back([&,tid](){
            std::uniform_real_distribution<double> coord(-kHalfBox,kHalfBox);
            for(std::uint64_t t=tid;t<trials;t+=nt){
                std::mt19937_64 gen(trial_seed(seed,t));
                std::vector<std::vector<Segment>> groups(static_cast<std::size_t>(max_rods));
                std::size_t total_fragments=0;
                for(int r=0;r<max_rods;++r){
                    const Vec3 c{coord(gen),coord(gen),coord(gen)};
                    const Vec3 d=random_isotropic_direction(gen);
                    groups[static_cast<std::size_t>(r)]=wrapped_rod_images(
                        {c-d*(kRodLength/2.0),c+d*(kRodLength/2.0),r,0});
                    total_fragments+=groups[static_cast<std::size_t>(r)].size();
                }
                const int left=static_cast<int>(total_fragments),right=left+1;
                DisjointSet dsu(total_fragments+2U);
                std::vector<Segment> active;
                active.reserve(total_fragments);
                bool conductive=false;
                for(int r=0;r<max_rods;++r){
                    int first_fragment=-1;
                    for(const Segment& fragment:groups[static_cast<std::size_t>(r)]){
                        const int node=static_cast<int>(active.size());
                        if(first_fragment<0) first_fragment=node;
                        else if(connectivity==FragmentConnectivity::SourceConnected)
                            dsu.unite(first_fragment,node);
                        const auto electrode=rod_electrode_contacts(fragment);
                        if(electrode.first) dsu.unite(node,left);
                        if(electrode.second) dsu.unite(node,right);
                        for(std::size_t j=0;j<active.size();++j){
                            if(active[j].source_id==fragment.source_id) continue;
                            if(!bbox_maybe_contact(fragment,active[j],kRodRodCutoff)) continue;
                            if(fragment_flat_cylinders_within_gap(fragment,active[j],kGap)){
                                dsu.unite(node,static_cast<int>(j));
                            }
                        }
                        active.push_back(fragment);
                    }
                    const int count=r+1;
                    if(!conductive && dsu.find(left)==dsu.find(right)) conductive=true;
                    if(conductive){
                        const int first_recorded=std::max(count,min_rods);
                        for(int n=first_recorded;n<=max_rods;++n){
                            ++local[tid][static_cast<std::size_t>(n-min_rods)];
                        }
                        break;
                    }
                }
            }
        });
    }
    for(auto& th:workers) th.join();
    std::vector<ProbabilityResult> out; out.reserve(nres);
    for(int n=min_rods;n<=max_rods;++n){
        std::uint64_t s=0; for(const auto& v:local) s+=v[static_cast<std::size_t>(n-min_rods)];
        out.push_back({n,0,s,trials});
    }
    return out;
}

struct CellKey { int x{},y{},z{}; bool operator==(const CellKey& o)const{return x==o.x&&y==o.y&&z==o.z;} };
struct CellKeyHash { std::size_t operator()(const CellKey& k)const noexcept{
    const std::uint64_t a=static_cast<std::uint64_t>(k.x+100000)*73856093ULL;
    const std::uint64_t b=static_cast<std::uint64_t>(k.y+100000)*19349663ULL;
    const std::uint64_t c=static_cast<std::uint64_t>(k.z+100000)*83492791ULL;
    return static_cast<std::size_t>(a^b^c);
}};

CellKey point_cell(const Vec3& point,double cell_size){
    return {
        static_cast<int>(std::floor(point.x/cell_size)),
        static_cast<int>(std::floor(point.y/cell_size)),
        static_cast<int>(std::floor(point.z/cell_size))
    };
}

struct TrialGraph {
    std::vector<Segment> rods;
    std::vector<SphereImage> spheres;
    std::vector<Edge> edges;
    std::vector<unsigned char> left,right;
};

TrialGraph build_trial_graph_from_sources(const std::vector<Segment>& source_rods,
                                          const std::vector<Vec3>& source_spheres,
                                          FragmentConnectivity connectivity){
    TrialGraph g;
    g.rods.reserve(source_rods.size()*2U);
    for(const Segment& rod:source_rods){
        const auto fragments=wrapped_rod_images(rod);
        g.rods.insert(g.rods.end(),fragments.begin(),fragments.end());
    }
    g.spheres.reserve(source_spheres.size()*2U);
    for(std::size_t source=0;source<source_spheres.size();++source){
        const auto fragments=wrapped_sphere_images(
            source_spheres[source],static_cast<int>(source));
        g.spheres.insert(g.spheres.end(),fragments.begin(),fragments.end());
    }
    const std::size_t nr=g.rods.size(), nb=g.spheres.size(), nodes=nr+nb;
    g.left.assign(nodes,0); g.right.assign(nodes,0);
    for(std::size_t i=0;i<nr;++i){
        const auto contacts=rod_electrode_contacts(g.rods[i]);
        g.left[i]=contacts.first;
        g.right[i]=contacts.second;
    }
    for(std::size_t i=0;i<nb;++i){
        const std::size_t n=nr+i;
        const auto contacts=sphere_electrode_contacts(g.spheres[i].center);
        g.left[n]=contacts.first;
        g.right[n]=contacts.second;
    }
    // rod-rod
    for(std::size_t i=0;i<nr;++i) for(std::size_t j=i+1;j<nr;++j){
        if(g.rods[i].source_id==g.rods[j].source_id) continue;
        if(!bbox_maybe_contact(g.rods[i],g.rods[j],kRodRodCutoff)) continue;
        if(fragment_flat_cylinders_within_gap(g.rods[i],g.rods[j],kGap))
            g.edges.push_back({static_cast<int>(i),static_cast<int>(j)});
    }
    // sphere-sphere: one-pass uniform grid, each prior fragment is checked once.
    std::unordered_map<CellKey,std::vector<std::size_t>,CellKeyHash> sphere_grid;
    sphere_grid.reserve(nb*2U+1U);
    for(std::size_t i=0;i<nb;++i){
        const CellKey cell=point_cell(g.spheres[i].center,kSphereSphereCutoff);
        for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz){
            const auto found=sphere_grid.find({cell.x+dx,cell.y+dy,cell.z+dz});
            if(found==sphere_grid.end()) continue;
            for(const std::size_t j:found->second){
                if(g.spheres[i].source_id==g.spheres[j].source_id) continue;
                if(fragment_spheres_within_gap(g.spheres[i].center,g.spheres[j].center,kGap))
                    g.edges.push_back({static_cast<int>(nr+j),static_cast<int>(nr+i)});
            }
        }
        sphere_grid[cell].push_back(i);
    }
    // rod-sphere: query the sphere grid over the rod capsule's conservative AABB.
    std::unordered_map<CellKey,std::vector<std::size_t>,CellKeyHash> mixed_sphere_grid;
    mixed_sphere_grid.reserve(nb*2U+1U);
    for(std::size_t b=0;b<nb;++b)
        mixed_sphere_grid[point_cell(g.spheres[b].center,kRodSphereCutoff)].push_back(b);
    for(std::size_t r=0;r<nr;++r){
        const Vec3 lower{
            std::min(g.rods[r].p.x,g.rods[r].q.x)-kRodSphereCutoff,
            std::min(g.rods[r].p.y,g.rods[r].q.y)-kRodSphereCutoff,
            std::min(g.rods[r].p.z,g.rods[r].q.z)-kRodSphereCutoff};
        const Vec3 upper{
            std::max(g.rods[r].p.x,g.rods[r].q.x)+kRodSphereCutoff,
            std::max(g.rods[r].p.y,g.rods[r].q.y)+kRodSphereCutoff,
            std::max(g.rods[r].p.z,g.rods[r].q.z)+kRodSphereCutoff};
        const CellKey first_cell=point_cell(lower,kRodSphereCutoff);
        const CellKey last_cell=point_cell(upper,kRodSphereCutoff);
        for(int x=first_cell.x;x<=last_cell.x;++x)
            for(int y=first_cell.y;y<=last_cell.y;++y)
                for(int z=first_cell.z;z<=last_cell.z;++z){
                    const auto found=mixed_sphere_grid.find({x,y,z});
                    if(found==mixed_sphere_grid.end()) continue;
                    for(const std::size_t b:found->second){
                        if(fragment_flat_cylinder_sphere_within_gap(
                               g.rods[r],g.spheres[b].center,kGap))
                            g.edges.push_back({static_cast<int>(r),static_cast<int>(nr+b)});
                    }
                }
    }
    if(connectivity==FragmentConnectivity::SourceConnected){
        std::vector<int> first_rod(source_rods.size(),-1);
        for(std::size_t i=0;i<nr;++i){
            int& first=first_rod[static_cast<std::size_t>(g.rods[i].source_id)];
            if(first<0) first=static_cast<int>(i);
            else g.edges.push_back({first,static_cast<int>(i)});
        }
        std::vector<int> first_sphere(source_spheres.size(),-1);
        for(std::size_t i=0;i<nb;++i){
            int& first=first_sphere[static_cast<std::size_t>(g.spheres[i].source_id)];
            const int node=static_cast<int>(nr+i);
            if(first<0) first=node;
            else g.edges.push_back({first,node});
        }
    }
    return g;
}

TrialGraph build_trial_graph(int max_rods,int max_spheres,std::uint64_t seed,
                             FragmentConnectivity connectivity=FragmentConnectivity::Independent){
    std::mt19937_64 rod_gen(seed);
    std::mt19937_64 sphere_gen(seed^0xd1b54a32d192ed03ULL);
    std::uniform_real_distribution<double> coord(-kHalfBox,kHalfBox);
    std::vector<Segment> source_rods;
    source_rods.reserve(static_cast<std::size_t>(max_rods));
    for(int r=0;r<max_rods;++r){
        const Vec3 c{coord(rod_gen),coord(rod_gen),coord(rod_gen)};
        const Vec3 d=random_isotropic_direction(rod_gen);
        source_rods.push_back({c-d*(kRodLength/2.0),c+d*(kRodLength/2.0),r,0});
    }
    std::vector<Vec3> source_spheres;
    source_spheres.reserve(static_cast<std::size_t>(max_spheres));
    for(int b=0;b<max_spheres;++b){
        source_spheres.push_back({coord(sphere_gen),coord(sphere_gen),coord(sphere_gen)});
    }
    return build_trial_graph_from_sources(source_rods,source_spheres,connectivity);
}

bool evaluate_candidate(const TrialGraph& g,const Candidate& c){
    const int nr=static_cast<int>(g.rods.size()), nb=static_cast<int>(g.spheres.size());
    const int left=nr+nb,right=left+1;
    DisjointSet dsu(static_cast<std::size_t>(right+1));
    auto active=[&](int n){
        if(n<nr) return g.rods[static_cast<std::size_t>(n)].source_id<c.rods;
        return g.spheres[static_cast<std::size_t>(n-nr)].source_id<c.spheres;
    };
    for(int n=0;n<left;++n){ if(!active(n)) continue; if(g.left[static_cast<std::size_t>(n)]) dsu.unite(n,left); if(g.right[static_cast<std::size_t>(n)]) dsu.unite(n,right); }
    for(const Edge& e:g.edges) if(active(e.first)&&active(e.second)) dsu.unite(e.first,e.second);
    return dsu.find(left)==dsu.find(right);
}

std::vector<ProbabilityResult> simulate_mixed(const std::vector<Candidate>& candidates,std::uint64_t trials,std::uint64_t seed,unsigned requested_threads,
                                              FragmentConnectivity connectivity=FragmentConnectivity::Independent){
    if(candidates.empty()||trials==0) throw std::invalid_argument("mixed simulation requires candidates and trials");
    int mr=0,mb=0; for(const auto& c:candidates){if(c.rods<0||c.spheres<0) throw std::invalid_argument("negative count"); mr=std::max(mr,c.rods); mb=std::max(mb,c.spheres);}
    const unsigned nt=thread_count(requested_threads,trials);
    std::vector<std::vector<std::uint64_t>> local(nt,std::vector<std::uint64_t>(candidates.size(),0));
    std::vector<std::thread> workers;
    for(unsigned tid=0;tid<nt;++tid){
        workers.emplace_back([&,tid](){
            for(std::uint64_t t=tid;t<trials;t+=nt){
                const TrialGraph g=build_trial_graph(mr,mb,trial_seed(seed,t),connectivity);
                for(std::size_t i=0;i<candidates.size();++i) if(evaluate_candidate(g,candidates[i])) ++local[tid][i];
            }
        });
    }
    for(auto& th:workers) th.join();
    std::vector<ProbabilityResult> out; out.reserve(candidates.size());
    for(std::size_t i=0;i<candidates.size();++i){std::uint64_t s=0;for(const auto& v:local)s+=v[i];out.push_back({candidates[i].rods,candidates[i].spheres,s,trials});}
    return out;
}

std::string path_for_message(const fs::path& path) {
#if defined(_WIN32)
    return path.u8string();
#else
    return path.string();
#endif
}

void normalize_csv_line(std::string& line) {
    if (line.size() >= 3U &&
        static_cast<unsigned char>(line[0]) == 0xEFU &&
        static_cast<unsigned char>(line[1]) == 0xBBU &&
        static_cast<unsigned char>(line[2]) == 0xBFU) {
        line.erase(0U, 3U);
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    for (char& ch : line) {
        if (ch == ',' || ch == ';' || ch == '\t') {
            ch = ' ';
        }
    }
}

void ensure_parent_directory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

std::vector<Candidate> read_candidates(const fs::path& path){
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open candidate file: "+path_for_message(path));
    std::vector<Candidate> out; std::string line;
    while(std::getline(in,line)){
        normalize_csv_line(line);
        const std::size_t first = line.find_first_not_of(" ");
        if(first == std::string::npos || line[first] == '#') continue;
        std::istringstream ss(line);Candidate c;
        if(ss>>c.rods>>c.spheres) out.push_back(c);
    }
    if (out.empty()) {
        throw std::runtime_error("no valid candidates in: "+path_for_message(path));
    }
    return out;
}

std::vector<Candidate> enumerate_strictly_cheaper_candidates(double upper_cost){
    if(!(upper_cost>kCostA+kCostB)) {
        throw std::invalid_argument("cost upper bound leaves no N_A>=1,N_B>=1 candidate");
    }
    std::vector<Candidate> out;
    for(int rods=1;kCostA*static_cast<double>(rods)+kCostB<upper_cost;++rods){
        for(int spheres=1;kCostA*static_cast<double>(rods)+
                          kCostB*static_cast<double>(spheres)<upper_cost;++spheres){
            out.push_back({rods,spheres});
        }
    }
    return out;
}

void write_candidates(const std::vector<Candidate>& candidates,const fs::path& path){
    ensure_parent_directory(path);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot write "+path_for_message(path));
    out<<"N_A,N_B,cost_yuan\n"<<std::setprecision(12);
    for(const auto& candidate:candidates){
        out<<candidate.rods<<','<<candidate.spheres<<','
           <<kCostA*static_cast<double>(candidate.rods)+
             kCostB*static_cast<double>(candidate.spheres)<<'\n';
    }
    if(!out) throw std::runtime_error("failed while writing "+path_for_message(path));
}
std::vector<Segment> read_segments(const fs::path& path){
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open segment file: "+path_for_message(path));
    std::vector<Segment> out; std::string line; int id=0;
    while(std::getline(in,line)){
        normalize_csv_line(line);
        std::istringstream ss(line);Segment segment;segment.source_id=id;
        if(ss>>segment.p.x>>segment.p.y>>segment.p.z>>segment.q.x>>segment.q.y>>segment.q.z){out.push_back(segment);++id;}
    }
    if (out.empty()) {
        throw std::runtime_error("no numeric segments in: "+path_for_message(path));
    }
    return out;
}

struct Q1Result {
    bool conductive{};
    std::size_t edges{};
    std::vector<int> path;
    std::vector<int> left_component;
    std::vector<int> right_component;
    double max_path_gap_nm{};
};
Q1Result solve_q1(const std::vector<Segment>& segs){
    const int n=static_cast<int>(segs.size()), left=n,right=n+1;
    DisjointSet dsu(static_cast<std::size_t>(n+2));
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n+2));
    std::size_t edges=0;
    auto add=[&](int a,int b){dsu.unite(a,b);adj[static_cast<std::size_t>(a)].push_back(b);adj[static_cast<std::size_t>(b)].push_back(a);};
    for(int i=0;i<n;++i){
        const auto range=flat_cylinder_x_range(segs[static_cast<std::size_t>(i)]);
        auto plane_distance=[&](double x){
            if(x<range.first) return range.first-x;
            if(x>range.second) return x-range.second;
            return 0.0;
        };
        if(plane_distance(-kHalfBox)<=kGap+1e-9)add(i,left);
        if(plane_distance( kHalfBox)<=kGap+1e-9)add(i,right);
    }
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j){
        if(!bbox_maybe_contact(segs[static_cast<std::size_t>(i)],segs[static_cast<std::size_t>(j)],kRodRodCutoff)) continue;
        if(flat_cylinders_within_gap(segs[static_cast<std::size_t>(i)],
                                     segs[static_cast<std::size_t>(j)],kGap)){
            add(i,j);++edges;
        }
    }
    Q1Result res;res.conductive=dsu.find(left)==dsu.find(right);res.edges=edges;
    for(int i=0;i<n;++i){
        if(dsu.find(i)==dsu.find(left)) res.left_component.push_back(i+1);
        if(dsu.find(i)==dsu.find(right)) res.right_component.push_back(i+1);
    }
    if(!res.conductive)return res;
    std::vector<int> prev(static_cast<std::size_t>(n+2),-1);std::queue<int> q;q.push(left);prev[static_cast<std::size_t>(left)]=left;
    while(!q.empty()){int u=q.front();q.pop();if(u==right)break;for(int v:adj[static_cast<std::size_t>(u)])if(prev[static_cast<std::size_t>(v)]<0){prev[static_cast<std::size_t>(v)]=u;q.push(v);}}
    for(int cur=right;cur!=left;cur=prev[static_cast<std::size_t>(cur)])res.path.push_back(cur);
    res.path.push_back(left);std::reverse(res.path.begin(),res.path.end());
    auto electrode_gap=[&](const Segment& rod,double plane){
        const auto range=flat_cylinder_x_range(rod);
        if(plane<range.first) return range.first-plane;
        if(plane>range.second) return plane-range.second;
        return 0.0;
    };
    for(std::size_t i=1;i<res.path.size();++i){
        const int a=res.path[i-1U],b=res.path[i];
        double gap=0.0;
        if(a==left) gap=electrode_gap(segs[static_cast<std::size_t>(b)],-kHalfBox);
        else if(b==right) gap=electrode_gap(segs[static_cast<std::size_t>(a)],kHalfBox);
        else gap=flat_cylinder_surface_gap(segs[static_cast<std::size_t>(a)],
                                           segs[static_cast<std::size_t>(b)],kGap);
        res.max_path_gap_nm=std::max(res.max_path_gap_nm,gap);
    }
    return res;
}

void write_q1_result(const Q1Result& result,std::size_t segment_count,const fs::path& path){
    ensure_parent_directory(path);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot write "+path_for_message(path));
    out<<"{\n  \"segments\": "<<segment_count
       <<",\n  \"contact_edges\": "<<result.edges
       <<",\n  \"conductive\": "<<(result.conductive?"true":"false")
       <<",\n  \"path\": [";
    for(std::size_t i=0;i<result.path.size();++i){
        if(i!=0U) out<<", ";
        if(result.path[i]==static_cast<int>(segment_count)) out<<"\"L\"";
        else if(result.path[i]==static_cast<int>(segment_count+1U)) out<<"\"R\"";
        else out<<(result.path[i]+1);
    }
    out<<"],\n  \"left_component\": [";
    for(std::size_t i=0;i<result.left_component.size();++i){
        if(i!=0U) out<<", ";
        out<<result.left_component[i];
    }
    out<<"],\n  \"right_component\": [";
    for(std::size_t i=0;i<result.right_component.size();++i){
        if(i!=0U) out<<", ";
        out<<result.right_component[i];
    }
    out<<"],\n  \"path_max_surface_gap_nm\": "<<std::setprecision(12)
       <<result.max_path_gap_nm<<"\n}\n";
    if(!out) throw std::runtime_error("failed while writing "+path_for_message(path));
}

std::pair<double,double> wilson(std::uint64_t s,std::uint64_t n){
    if (n == 0) {
        return {0.0, 0.0};
    }
    const double z=1.959963984540054,p=static_cast<double>(s)/static_cast<double>(n),nn=static_cast<double>(n);
    const double den=1+z*z/nn,ctr=(p+z*z/(2*nn))/den,half=z*std::sqrt(p*(1-p)/nn+z*z/(4*nn*nn))/den;
    double low=std::max(0.0,ctr-half), high=std::min(1.0,ctr+half);
    if(s==0U) low=0.0;
    if(s==n) high=1.0;
    return {low,high};
}
void write_prob(const std::vector<ProbabilityResult>& results,const fs::path& path){
    ensure_parent_directory(path);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot write "+path_for_message(path));
    out<<"N_A,N_B,successes,trials,probability,standard_error,Wilson95_low,Wilson95_high\n"<<std::setprecision(12);
    for(const auto& result:results){
        const double probability=static_cast<double>(result.successes)/static_cast<double>(result.trials);
        const double standard_error=std::sqrt(probability*(1.0-probability)/static_cast<double>(result.trials));
        const auto interval=wilson(result.successes,result.trials);
        out<<result.rods<<','<<result.spheres<<','<<result.successes<<','<<result.trials<<','
           <<probability<<','<<standard_error<<','<<interval.first<<','<<interval.second<<'\n';
    }
    if(!out) throw std::runtime_error("failed while writing "+path_for_message(path));
}

std::vector<std::uint64_t> first_conduction_histogram(
    const std::vector<ProbabilityResult>& prefix) {
    if(prefix.empty() || prefix.front().rods!=1) {
        throw std::invalid_argument("first-conduction histogram requires A-prefix starting at 1");
    }
    const std::uint64_t trials=prefix.front().trials;
    std::vector<std::uint64_t> counts(prefix.size()+1U,0U);
    std::uint64_t previous=0U;
    for(std::size_t i=0;i<prefix.size();++i){
        if(prefix[i].rods!=static_cast<int>(i+1U) || prefix[i].trials!=trials ||
           prefix[i].successes<previous){
            throw std::invalid_argument("invalid cumulative A-prefix for first-conduction histogram");
        }
        counts[i]=prefix[i].successes-previous;
        previous=prefix[i].successes;
    }
    if(previous>trials) throw std::invalid_argument("A-prefix successes exceed trials");
    counts.back()=trials-previous;
    return counts;
}

void write_first_conduction(const std::vector<ProbabilityResult>& prefix,const fs::path& path){
    const auto counts=first_conduction_histogram(prefix);
    const std::uint64_t trials=prefix.front().trials;
    ensure_parent_directory(path);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot write "+path_for_message(path));
    out<<"tau_A,count,relative_frequency,censored\n"<<std::setprecision(12);
    for(std::size_t i=0;i<prefix.size();++i){
        out<<(i+1U)<<','<<counts[i]<<','
           <<static_cast<double>(counts[i])/static_cast<double>(trials)<<",false\n";
    }
    out<<'>'<<prefix.back().rods<<','<<counts.back()<<','
       <<static_cast<double>(counts.back())/static_cast<double>(trials)<<",true\n";
    if(!out) throw std::runtime_error("failed while writing "+path_for_message(path));
}

void require(bool ok,const std::string& name){if(!ok)throw std::runtime_error("selftest failed: "+name);std::cout<<"PASS "<<name<<'\n';}
void selftest(){
    Segment a{{0,0,0},{10,0,0},0},b{{0,5,0},{10,5,0},1},c{{5,-5,0},{5,5,0},2};
    require(std::abs(segment_segment_distance2(a,b)-25.0)<1e-9,"parallel segment distance");
    require(segment_segment_distance2(a,c)<1e-12,"crossing segment distance");
    require(std::abs(point_segment_distance2({5,3,0},a)-9.0)<1e-9,"point-segment distance");
    const Segment side_only_overflow{{4980,-2500,0},{4980,2500,0},0};
    const auto side_only_x=flat_cylinder_axis_range(side_only_overflow,0);
    require(side_only_x.first> -kHalfBox && side_only_x.second>kHalfBox,
            "cylinder radius detects side-only boundary overflow");
    const auto side_only_contact=rod_electrode_contacts(side_only_overflow);
    require(side_only_contact.first==0U && side_only_contact.second==1U,
            "side-only overflow directly contacts only the right electrode");
    const Segment corner_overflow{{4980,4980,-2500},{4980,4980,2500},0};
    const auto corner_x=flat_cylinder_axis_range(corner_overflow,0);
    const auto corner_y=flat_cylinder_axis_range(corner_overflow,1);
    require(corner_x.second>kHalfBox && corner_y.second>kHalfBox,
            "cylinder radius detects simultaneous two-face overflow");
    require(wrapped_sphere_images({4900,4900,4900},0).size()==8U,"sphere corner image count");
    const Segment periodic_left{{-4999.5,0,0},{-3000,0,0},0};
    const Segment periodic_right{{3000,0,0},{4999.5,0,0},1};
    require(!periodic_flat_cylinders_within_gap(periodic_left,periodic_right,kGap),
            "non-crossing opposite A cylinders do not receive periodic contact");
    require(!flat_cylinders_within_gap(periodic_left,periodic_right,kGap),
            "periodic A-A contact requires translated image");
    const Segment crossed_left{{-5000.5,0,0},{-3000,0,0},0};
    const Segment crossed_right{{3000,0,0},{5000.5,0,0},1};
    require(periodic_flat_cylinders_within_gap(crossed_left,crossed_right,kGap),
            "actually crossed A-A fragments can contact");
    require(periodic_flat_cylinder_sphere_within_gap(
                crossed_left,{4800.0,0.0,0.0},kGap),
            "actually crossed A fragment can contact B");
    require(periodic_spheres_within_gap({-4900.0,0.0,0.0},{4900.0,0.0,0.0},kGap),
            "actually crossed B fragments can contact");
    TrialGraph wrapped_single_rod;
    wrapped_single_rod.rods={{{3500,0,0},{6000,0,0},0}};
    wrapped_single_rod.left={1U};
    wrapped_single_rod.right={1U};
    require(evaluate_candidate(wrapped_single_rod,{1,0}),
            "manual graph node attached to both electrodes conducts");
    const std::vector<Segment> no_source_rods;
    const std::vector<Vec3> one_crossing_source_sphere{{4900.0,0.0,0.0}};
    const auto independent_crossing_graph=build_trial_graph_from_sources(
        no_source_rods,one_crossing_source_sphere,FragmentConnectivity::Independent);
    const auto connected_crossing_graph=build_trial_graph_from_sources(
        no_source_rods,one_crossing_source_sphere,FragmentConnectivity::SourceConnected);
    require(!evaluate_candidate(independent_crossing_graph,{0,1}),
            "D mode keeps actual crossing fragments electrically separate");
    require(evaluate_candidate(connected_crossing_graph,{0,1}),
            "S mode connects only fragments of the actual crossing source");
    const TrialGraph small_graph=build_trial_graph(1,1,987654321ULL);
    const TrialGraph large_graph=build_trial_graph(4,5,987654321ULL);
    require(norm2(small_graph.rods[0].p-large_graph.rods[0].p)<1e-20 &&
            norm2(small_graph.rods[0].q-large_graph.rods[0].q)<1e-20 &&
            norm2(small_graph.spheres[0].center-large_graph.spheres[0].center)<1e-20,
            "A/B random streams independent of candidate maxima");
    const Segment aa_false_positive_a{{-2500,0,0},{2500,0,0},0};
    const Segment aa_false_positive_b{{2510,60,0},{7510,60,0},1};
    require(!flat_cylinders_within_gap(aa_false_positive_a,aa_false_positive_b,kGap),
            "flat cylinders reject capsule-only A-A contact");
    const Segment ab_false_positive_rod{{-2500,0,0},{2500,0,0},0};
    const Vec3 ab_false_positive_sphere_center{2700,100,0};
    require(!flat_cylinder_sphere_within_gap(ab_false_positive_rod,
                                              ab_false_positive_sphere_center,kGap),
            "flat cylinder rejects capsule-only A-B contact");
    const auto exact_x_range=flat_cylinder_x_range(ab_false_positive_rod);
    require(std::abs(exact_x_range.second-2500.0)<1e-9,
            "flat cylinder uses exact x projection");
    const Segment aa_true_contact{{2501,60,0},{7501,60,0},1};
    require(flat_cylinders_within_gap(aa_false_positive_a,aa_true_contact,kGap),
            "flat cylinder A-A cap-rim contact");
    require(flat_cylinders_within_gap(aa_true_contact,aa_false_positive_a,kGap),
            "flat cylinder A-A contact symmetry");
    require(flat_cylinders_within_gap(aa_false_positive_a,aa_false_positive_a,kGap),
            "flat cylinder A-A overlap");
    require(flat_cylinder_sphere_within_gap(ab_false_positive_rod,{2701,0,0},kGap),
            "flat cylinder A-B cap contact");
    require(flat_cylinder_sphere_within_gap(ab_false_positive_rod,{0,231,0},kGap),
            "flat cylinder A-B side contact");
    require(flat_cylinder_sphere_within_gap(ab_false_positive_rod,{0,0,0},kGap),
            "flat cylinder A-B overlap");
    const Segment y_axis_rod{{0,-2500,0},{0,2500,0},0};
    const auto y_axis_range=flat_cylinder_x_range(y_axis_rod);
    require(std::abs(y_axis_range.first+30.0)<1e-9 &&
            std::abs(y_axis_range.second-30.0)<1e-9,
            "flat cylinder transverse x projection");
    require(solve_q1({aa_false_positive_a,aa_false_positive_b}).edges==0U,
            "Q1 rejects capsule-only A-A edge");
    std::vector<Segment> disconnected{{{-5000,0,0},{-4000,0,0},0},{{4000,0,0},{5000,0,0},1}};
    require(!solve_q1(disconnected).conductive,"manual disconnected network");
    std::vector<Segment> connected{{{-5000,0,0},{-1000,0,0},0},{{-999,0,0},{3000,0,0},1},{{3001,0,0},{5000,0,0},2}};
    const auto connected_result=solve_q1(connected);
    require(connected_result.conductive,"manual connected network");
    require(std::abs(connected_result.max_path_gap_nm-1.0)<1e-6,
            "Q1 reports maximum path surface gap");

    const auto prefix_one=simulate_a_prefix(5,8,20,123,1);
    const auto prefix_many=simulate_a_prefix(5,8,20,123,4);
    require(prefix_one.size()==prefix_many.size(),"thread output size");
    for(std::size_t i=0;i<prefix_one.size();++i){
        require(prefix_one[i].successes==prefix_many[i].successes,"A-prefix thread-independent reproducibility");
    }
    for(std::size_t i=1;i<prefix_one.size();++i){
        require(prefix_one[i].successes>=prefix_one[i-1].successes,"prefix monotonicity");
    }

    const std::vector<Candidate> mixed_candidates{{5,5},{6,3},{8,0}};
    const auto mixed_one=simulate_mixed(mixed_candidates,20,456,1);
    const auto mixed_many=simulate_mixed(mixed_candidates,20,456,4);
    require(mixed_one.size()==mixed_many.size(),"mixed thread output size");
    for(std::size_t i=0;i<mixed_one.size();++i){
        require(mixed_one[i].successes==mixed_many[i].successes,"mixed thread-independent reproducibility");
    }
    const auto mixed_small_domain=simulate_mixed({Candidate{1,5}},200,789,2);
    const auto mixed_large_domain=simulate_mixed({Candidate{1,5},Candidate{6,50}},200,789,2);
    require(mixed_small_domain[0].successes==mixed_large_domain[0].successes,
            "mixed candidate result independent of domain maxima");

    const auto interval=wilson(90,100);
    require(interval.first>0.82 && interval.first<0.83 && interval.second>0.94 && interval.second<0.95,
            "Wilson interval reference case");
    require(wilson(0,100).first==0.0 && wilson(100,100).second==1.0,
            "Wilson endpoint clamping");
    const auto tau_counts=first_conduction_histogram({
        ProbabilityResult{1,0,2,10},ProbabilityResult{2,0,5,10},
        ProbabilityResult{3,0,5,10}});
    require(tau_counts==std::vector<std::uint64_t>({2,3,0,5}),
            "first-conduction histogram from cumulative prefix");
    const auto cheap=enumerate_strictly_cheaper_candidates(kCostA+3.0*kCostB);
    require(cheap.size()==2U && cheap[0].rods==1 && cheap[0].spheres==1 &&
            cheap[1].rods==1 && cheap[1].spheres==2,
            "strict Q4 cost-domain enumeration");

    const fs::path temp_root=fs::temp_directory_path()/fs::u8path("hscup_unicode_路径测试");
    std::error_code cleanup_error;
    fs::remove_all(temp_root,cleanup_error);
    fs::create_directories(temp_root);
    const fs::path segment_path=temp_root/fs::u8path("组1_坐标.csv");
    {
        std::ofstream out(segment_path,std::ios::binary);
        out<<"\xEF\xBB\xBFx1,y1,z1,x2,y2,z2\n"
           <<"-5000,0,0,-1000,0,0\n"
           <<"-950,0,0,3000,0,0\n"
           <<"3050,0,0,5000,0,0\n";
    }
    require(read_segments(segment_path).size()==3U,"Unicode path and UTF-8 BOM input");
    const fs::path nested_output=temp_root/fs::u8path("结果/快速复核.csv");
    write_prob({ProbabilityResult{1,0,1,1}},nested_output);
    require(fs::exists(nested_output),"automatic output directory creation");
    fs::remove_all(temp_root,cleanup_error);
}

std::string argument_text(const fs::path& argument){return path_for_message(argument);}

int parse_int_argument(const fs::path& argument,const char* name){
    const std::string text=argument_text(argument);
    std::size_t used=0U;
    const long long value=std::stoll(text,&used,10);
    if(used!=text.size() || value<std::numeric_limits<int>::min() || value>std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string("invalid ")+name+": "+text);
    return static_cast<int>(value);
}
std::uint64_t parse_u64_argument(const fs::path& argument,const char* name){
    const std::string text=argument_text(argument);
    if(!text.empty() && text.front()=='-') throw std::invalid_argument(std::string("invalid ")+name+": "+text);
    std::size_t used=0U;
    const unsigned long long value=std::stoull(text,&used,10);
    if(used!=text.size()) throw std::invalid_argument(std::string("invalid ")+name+": "+text);
    return static_cast<std::uint64_t>(value);
}
double parse_double_argument(const fs::path& argument,const char* name){
    const std::string text=argument_text(argument);
    std::size_t used=0U;
    const double value=std::stod(text,&used);
    if(used!=text.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string("invalid ")+name+": "+text);
    return value;
}
unsigned parse_thread_argument(const fs::path& argument){
    const std::uint64_t value=parse_u64_argument(argument,"threads");
    if(value>static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()))
        throw std::invalid_argument("threads exceeds platform limit");
    return static_cast<unsigned>(value);
}
FragmentConnectivity parse_fragment_connectivity(const fs::path& argument){
    const std::string value=argument_text(argument);
    if(value=="fragment-independent" || value=="D")
        return FragmentConnectivity::Independent;
    if(value=="source-connected" || value=="S")
        return FragmentConnectivity::SourceConnected;
    throw std::invalid_argument(
        "fragment connectivity must be fragment-independent|source-connected");
}
const char* fragment_connectivity_name(FragmentConnectivity connectivity){
    return connectivity==FragmentConnectivity::Independent
        ? "fragment-independent" : "source-connected";
}

void print_usage(){
    std::cout
        <<"microstructure_sim "<<kProgramVersion<<"\n"
        <<"Usage:\n"
        <<"  microstructure_sim selftest\n"
        <<"  microstructure_sim info\n"
        <<"  microstructure_sim q1 <segments.csv>\n"
        <<"  microstructure_sim q1-report <segments.csv> <result.json>\n"
        <<"  microstructure_sim a-prefix <min_A> <max_A> <trials> <seed> <output.csv> [threads] [fragment-independent|source-connected]\n"
        <<"  microstructure_sim a-study <max_A> <trials> <seed> <probabilities.csv> <first_conduction.csv> [threads] [fragment-independent|source-connected]\n"
        <<"  microstructure_sim cost-domain <strict_upper_cost_yuan> <candidates.csv>\n"
        <<"  microstructure_sim search-domain <feasible_N_A> <feasible_N_B> <candidates.csv>\n"
        <<"  microstructure_sim mixed <candidates.csv> <trials> <seed> <output.csv> [threads] [fragment-independent|source-connected]\n";
}

void print_info(){
    std::cout<<"version="<<kProgramVersion<<'\n';
#if defined(_WIN32)
    std::cout<<"platform=Windows\n";
#elif defined(__APPLE__)
    std::cout<<"platform=macOS\n";
#elif defined(__linux__)
    std::cout<<"platform=Linux\n";
#else
    std::cout<<"platform=unknown\n";
#endif
#if defined(_MSC_VER)
    std::cout<<"compiler=MSVC-compatible "<<_MSC_VER<<'\n';
#elif defined(__clang__)
    std::cout<<"compiler=Clang "<<__clang_major__<<'.'<<__clang_minor__<<'\n';
#elif defined(__GNUC__)
    std::cout<<"compiler=GCC "<<__GNUC__<<'.'<<__GNUC_MINOR__<<'\n';
#endif
    std::cout<<std::setprecision(12)
             <<"box_length_nm="<<kBoxLength<<'\n'
             <<"rod_length_nm="<<kRodLength<<'\n'
             <<"rod_radius_nm="<<kRodRadius<<'\n'
             <<"max_rod_axis_projection_halfwidth_nm="
             <<std::sqrt((kRodLength*kRodLength/4.0)+kRodRadius*kRodRadius)<<'\n'
             <<"sphere_radius_nm="<<kSphereRadius<<'\n'
             <<"gap_nm="<<kGap<<'\n'
             <<"default_fragment_connectivity="
             <<fragment_connectivity_name(FragmentConnectivity::Independent)<<'\n';
}

int run_cli(const std::vector<fs::path>& arguments){
    try{
        if(arguments.size()<2U){print_usage();return 2;}
        const std::string command=argument_text(arguments[1]);
        if(command=="--help" || command=="-h" || command=="help"){print_usage();return 0;}
        if(command=="--version" || command=="version"){std::cout<<kProgramVersion<<'\n';return 0;}
        if(command=="info"){print_info();return 0;}
        if(command=="selftest"){selftest();return 0;}
        if(command=="q1"){
            if(arguments.size()<3U)throw std::invalid_argument("q1 requires a CSV path");
            const auto segments=read_segments(arguments[2]);const auto result=solve_q1(segments);
            std::cout<<"segments="<<segments.size()<<" contact_edges="<<result.edges<<" conductive="<<(result.conductive?"yes":"no")<<"\n";
            if(result.conductive){
                std::cout<<"path=";
                for(std::size_t i=0;i<result.path.size();++i){
                    if(i!=0U)std::cout<<"->";
                    if(result.path[i]==static_cast<int>(segments.size()))std::cout<<"L";
                    else if(result.path[i]==static_cast<int>(segments.size()+1U))std::cout<<"R";
                    else std::cout<<result.path[i]+1;
                }
                std::cout<<"\npath_max_surface_gap_nm="<<std::setprecision(12)
                         <<result.max_path_gap_nm<<"\n";
            }
            return 0;
        }
        if(command=="a-prefix"){
            if(arguments.size()<7U)throw std::invalid_argument("a-prefix min max trials seed output.csv [threads]");
            const int min_rods=parse_int_argument(arguments[2],"min_A");
            const int max_rods=parse_int_argument(arguments[3],"max_A");
            const std::uint64_t trials=parse_u64_argument(arguments[4],"trials");
            const std::uint64_t seed=parse_u64_argument(arguments[5],"seed");
            const unsigned threads=arguments.size()>=8U?parse_thread_argument(arguments[7]):0U;
            const auto connectivity=arguments.size()>=9U
                ?parse_fragment_connectivity(arguments[8])
                :FragmentConnectivity::Independent;
            write_prob(simulate_a_prefix(min_rods,max_rods,trials,seed,threads,connectivity),arguments[6]);
            return 0;
        }
        if(command=="q1-report"){
            if(arguments.size()<4U)throw std::invalid_argument("q1-report requires segments.csv result.json");
            const auto segments=read_segments(arguments[2]);
            write_q1_result(solve_q1(segments),segments.size(),arguments[3]);
            return 0;
        }
        if(command=="a-study"){
            if(arguments.size()<7U)throw std::invalid_argument("a-study max trials seed probabilities.csv first_conduction.csv [threads]");
            const int max_rods=parse_int_argument(arguments[2],"max_A");
            const std::uint64_t trials=parse_u64_argument(arguments[3],"trials");
            const std::uint64_t seed=parse_u64_argument(arguments[4],"seed");
            const unsigned threads=arguments.size()>=8U?parse_thread_argument(arguments[7]):0U;
            const auto connectivity=arguments.size()>=9U
                ?parse_fragment_connectivity(arguments[8])
                :FragmentConnectivity::Independent;
            const auto prefix=simulate_a_prefix(1,max_rods,trials,seed,threads,connectivity);
            write_prob(prefix,arguments[5]);
            write_first_conduction(prefix,arguments[6]);
            return 0;
        }
        if(command=="mixed"){
            if(arguments.size()<6U)throw std::invalid_argument("mixed candidates.csv trials seed output.csv [threads]");
            const auto candidates=read_candidates(arguments[2]);
            const std::uint64_t trials=parse_u64_argument(arguments[3],"trials");
            const std::uint64_t seed=parse_u64_argument(arguments[4],"seed");
            const unsigned threads=arguments.size()>=7U?parse_thread_argument(arguments[6]):0U;
            const auto connectivity=arguments.size()>=8U
                ?parse_fragment_connectivity(arguments[7])
                :FragmentConnectivity::Independent;
            write_prob(simulate_mixed(candidates,trials,seed,threads,connectivity),arguments[5]);
            return 0;
        }
        if(command=="cost-domain"){
            if(arguments.size()<4U)throw std::invalid_argument("cost-domain strict_upper_cost_yuan candidates.csv");
            const double upper=parse_double_argument(arguments[2],"strict_upper_cost_yuan");
            write_candidates(enumerate_strictly_cheaper_candidates(upper),arguments[3]);
            return 0;
        }
        if(command=="search-domain"){
            if(arguments.size()<5U)throw std::invalid_argument("search-domain feasible_N_A feasible_N_B candidates.csv");
            const Candidate incumbent{parse_int_argument(arguments[2],"feasible_N_A"),
                                      parse_int_argument(arguments[3],"feasible_N_B")};
            if(incumbent.rods<1 || incumbent.spheres<1)
                throw std::invalid_argument("search-domain requires N_A>=1,N_B>=1");
            const double upper=kCostA*static_cast<double>(incumbent.rods)+
                               kCostB*static_cast<double>(incumbent.spheres);
            auto candidates=enumerate_strictly_cheaper_candidates(upper);
            candidates.push_back(incumbent);
            write_candidates(candidates,arguments[4]);
            return 0;
        }
        throw std::invalid_argument("unknown command: "+command);
    }catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<'\n';return 1;}
}

} // namespace model

#if defined(_WIN32) || defined(HSCUP_WIDE_ENTRY_TEST)
int wmain(int argc,wchar_t** argv){
    std::vector<std::filesystem::path> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for(int i=0;i<argc;++i) arguments.emplace_back(argv[i]);
    return model::run_cli(arguments);
}
#else
int main(int argc,char** argv){
    std::vector<std::filesystem::path> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for(int i=0;i<argc;++i) arguments.emplace_back(argv[i]);
    return model::run_cli(arguments);
}
#endif
