/**
 * @file pmm_trajectory3d.cpp
 * @author Robert Penicka (penicrob@fel.cvut.cz), Krystof Teissing (k.teissing@gmail.com)
 * @version 0.1
 * @date 2023-05-25
 * 
 */

#include "pmm_trajectory3d.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

namespace pmm {

PointMassTrajectory3D::PointMassTrajectory3D() {}
/*
Used by PMM-MG-Trajectory
version that converges to thrust limit by iterative increasing scaled (to mach
time) to the acc norm
this version scales time by default
*/
PointMassTrajectory3D::PointMassTrajectory3D(const QuadState &from, const QuadState &to,
                        const Scalar max_acc_norm, const Scalar v_max, const int max_iter,
                        const double precision_acc_limit, const bool debug) {
  // B = T + GVEC , |T|=max_acc_norm
  // initially B equal per axis with b_x=b_y=b_z -> |B-GVEC|^2 = |T|^2
  // -> 3*bs_x^2 + 2*g*a_x + g^2 - |T|^2 = 0 --> roots are the possible acc

  // const Scalar precision_acc_limit = 0.1;
  v_max_ = v_max;

  const Scalar max_acc_norm_pow2 = max_acc_norm * max_acc_norm;
  const Scalar a_equal_acc = 3;
  const Scalar b_equal_acc = 2 * G;
  const Scalar c_equal_acc = G * G - max_acc_norm_pow2;
  const Scalar equal_acc_1 = (-b_equal_acc + sqrt(b_equal_acc * b_equal_acc - 4 * a_equal_acc * c_equal_acc)) / (2 * a_equal_acc);

  Vector<3> per_axis_acc_vec;
  if ((Vector<3>::Constant(-equal_acc_1) - GVEC).norm() - max_acc_norm < PRECISION_PMM_VALUES) {
    per_axis_acc_vec = Vector<3>::Constant(equal_acc_1);
  } else {
    const Scalar equal_acc_2 =
      (-b_equal_acc - sqrt(b_equal_acc * b_equal_acc - 4 * a_equal_acc * c_equal_acc)) / (2 * a_equal_acc);
    per_axis_acc_vec = Vector<3>::Constant(equal_acc_2);
    std::cout << "that is happening ever? " << (Vector<3>::Constant(-equal_acc_1) - GVEC).norm() << " " << max_acc_norm << " " << (Vector<3>::Constant(-equal_acc_1) - GVEC).norm() - max_acc_norm << std::endl;
  }

    Vector<3> max_per_axis_acc_vec = per_axis_acc_vec;
  Vector<3> min_per_axis_acc_vec = -per_axis_acc_vec + 2*GVEC;


  PointMassTrajectory3D pmm3d(from, to, max_per_axis_acc_vec, min_per_axis_acc_vec, v_max_, true, true);
  
  Vector<3> start_acc = pmm3d.start_acc();
  Vector<3> end_acc = pmm3d.end_acc();
  Scalar start_thrust = (start_acc - GVEC).norm();
  Scalar end_thrust = (end_acc - GVEC).norm();

  Vector<3> biggest_acc = start_thrust > end_thrust ? start_acc : end_acc;
  Scalar largest_thrust = std::max(start_thrust, end_thrust);

  
  int iter = 0;
  while (fabs(largest_thrust - max_acc_norm) > precision_acc_limit and
         iter < max_iter) {
    iter++;
    // B = T + GVEC , |T|=max_acc_norm, |B-GVEC|^2 = |T|^2
    // scale the T parts by same factor k ->
    // k^2*b_x^2 + k^2*b_y^2 + (k*b_z + g)^2 - |T|^2 = 0 -> find k
    // (b_x^2 + b_y^2 + b_z^2) * k^2 + (2*g*b_z)*k

    const Scalar a_dis = biggest_acc.squaredNorm();
    const Scalar b_dis = 2 * biggest_acc(2) * G;
    const Scalar c_dis = G * G - (max_acc_norm * max_acc_norm);
    const Scalar k1 = (-b_dis + sqrt(b_dis * b_dis - 4 * a_dis * c_dis)) / (2 * a_dis);
    const Scalar k2 = (-b_dis - sqrt(b_dis * b_dis - 4 * a_dis * c_dis)) / (2 * a_dis);
    Vector<3> thrust_acc_new_k1 = k1 * biggest_acc - GVEC;
    Vector<3> thrust_acc_new_k2 = k2 * biggest_acc - GVEC;
    Vector<3> max_acc_new1 = thrust_acc_new_k1 + GVEC;
    Vector<3> max_acc_new2 = -thrust_acc_new_k1 + GVEC;

    if (debug) {
      std::cout << std::endl << "Iter: " << iter << std::endl;
      std::cout << "Thrust: " << largest_thrust << std::endl;
      std::cout << "max_acc_new1: " << max_acc_new1[0] << "," << max_acc_new1[1] << ","<< max_acc_new1[2] << std::endl;
      std::cout << "max_acc_new2: " << max_acc_new2[0] << "," << max_acc_new2[1] << ","<< max_acc_new2[2] << std::endl;
    }

    PointMassTrajectory3D pmm3d_new(from, to, max_acc_new1, max_acc_new2, v_max_, true, true);

    // pmm3d = PointMassTrajectory3D(from, to, max_acc_new1, max_acc_new2, v_max_, true, true);
    // pmm3d = pmm3d_new;

    if (debug){
      // std::cout << "T new: " << pmm3d.time() << std::endl;
      std::cout << "T new: " << pmm3d_new.time() << std::endl;
    }

    if (debug) {
      Vector<3> start_body_acc = (pmm3d_new.start_acc() - GVEC);
      Vector<3> end_body_acc = (pmm3d_new.end_acc() - GVEC);
      std::cout << std::endl << "Start thrust: " << (pmm3d_new.start_acc() - GVEC).norm() << std::endl;
      std::cout <<  "Start body acc: " << start_body_acc[0] << "," << start_body_acc[1] << ","<< start_body_acc[2] << std::endl;
      std::cout << "End thrust: " << (pmm3d_new.end_acc() - GVEC).norm() << std::endl;
      std::cout <<  "End body acc: " << end_body_acc[0] << "," << end_body_acc[1] << ","<< end_body_acc[2] << std::endl;
    }

    start_acc = pmm3d_new.start_acc();
    end_acc = pmm3d_new.end_acc();
    start_thrust = (start_acc - GVEC).norm();
    end_thrust = (end_acc - GVEC).norm();

    if (start_thrust > end_thrust) {
      biggest_acc = start_acc;
      largest_thrust = start_thrust;
    } else {
      biggest_acc = end_acc;
      largest_thrust = end_thrust;
    }


    x_ = pmm3d_new.x_;
    y_ = pmm3d_new.y_;
    z_ = pmm3d_new.z_;

    min_axis_trajectory_duration_ = pmm3d_new.min_axis_trajectory_duration_;
  }

  if (debug){
    std::cout << "Iter: " << iter << std::endl;
    std::cout << "Thrust: " << largest_thrust << std::endl;
  }

  // x_ = pmm3d.x_;
  // y_ = pmm3d.y_;
  // z_ = pmm3d.z_;

  // min_axis_trajectory_duration_ = pmm3d.min_axis_trajectory_duration_;
}

/*
basic version with symmetric acc limits in all axis using sync segments
*/
// Creates symmetrical acc limits and calls overladed function
PointMassTrajectory3D::PointMassTrajectory3D(const QuadState &from,
                                             const QuadState &to,
                                             const Vector<3> max_acc,
                                             const Scalar v_max,
                                             const bool equalize_time,
                                             const bool calc_gradient)
  : PointMassTrajectory3D(from, to, max_acc, -max_acc, equalize_time, calc_gradient, v_max) {}

PointMassTrajectory3D::PointMassTrajectory3D(const QuadState &from,
                                             const QuadState &to,
                                             const Vector<3> max_acc1,
                                             const Vector<3> max_acc2,
                                             const Scalar v_max,
                                             const bool equalize_time,
                                             const bool calc_gradient) {
  v_max_ = v_max;
  
  x_ = PMMTrajectory(from.p(0), from.v(0), to.p(0), to.v(0), max_acc1(0),
                     max_acc2(0), v_max_, 0, 0.0, false, calc_gradient, false);
  y_ = PMMTrajectory(from.p(1), from.v(1), to.p(1), to.v(1), max_acc1(1),
                     max_acc2(1), v_max_, 1, 0.0, false, calc_gradient, false);
  z_ = PMMTrajectory(from.p(2), from.v(2), to.p(2), to.v(2), max_acc1(2),
                     max_acc2(2), v_max_, 2, 0.0, false, calc_gradient, false);

  // keep track of min trajectory duration for each axis
  min_axis_trajectory_duration_[0] = x_.time();
  min_axis_trajectory_duration_[1] = y_.time();
  min_axis_trajectory_duration_[2] = z_.time();

  if (equalize_time && x_.exists_ && y_.exists_ && z_.exists_) {
    // compute sync trajectories according to bang-bang or bang-singular-bang approach
    Scalar tr_time = time();
    for(int i = 0; i<3; i++){
      auto axis_tr = get_axis_trajectory(i);
      if (fabs(axis_tr.time() - tr_time) > PRECISION_PMM_VALUES){
        // recomputation needed
        Scalar res = axis_tr.computeSyncTrajectory(tr_time, max_acc1(i), max_acc2(i), v_max);
        if (fabs(res) > PRECISION_PMM_VALUES){
          if (res > tr_time){
            // set new trajectory time and recompute all axis
            tr_time = res;
            min_axis_trajectory_duration_[i] = tr_time;
            set_axis_trajectory(i, axis_tr);
            i = -1;
            continue;
          } else {
            axis_tr.exists_=false;
          }
        }
        set_axis_trajectory(i, axis_tr);
      }
    }
  }
}

Vector<3> PointMassTrajectory3D::acc_in_time(const Scalar time_in_tr) const {
  Vector<3> acc;
  acc[0] = x_.acc_in_time(time_in_tr);
  acc[1] = y_.acc_in_time(time_in_tr);
  acc[2] = z_.acc_in_time(time_in_tr);
  return acc;
}

Vector<3> PointMassTrajectory3D::start_acc() const {
  return Vector<3>(x_.a_(0), y_.a_(0), z_.a_(0));
}

Vector<3> PointMassTrajectory3D::end_acc() const {
  return Vector<3>(x_.a_(1), y_.a_(1), z_.a_(1));
}

void PointMassTrajectory3D::set_axis_trajectory(const int i,
                                                const PMMTrajectory tr) {
  switch (i) {
    case 0:
      x_ = tr;
      break;
    case 1:
      y_ = tr;
      break;
    case 2:
      z_ = tr;
      break;
    default:
      std::cout << "bad axis index " << i << std::endl;
  }
}
PMMTrajectory &PointMassTrajectory3D::get_axis_trajectory(const int i) {
  switch (i) {
    case 0:
      return x_;
    case 1:
      return y_;
    case 2:
      return z_;
    default:
      std::cout << "bad axis index " << i << std::endl;
      return x_;
  }
}

Scalar PointMassTrajectory3D::get_axis_switch_time(const int i) const {
  switch (i) {
    case 0:
      return x_.t_(0);
    case 1:
      return y_.t_(0);
    case 2:
      return z_.t_(0);
    default:
      exit(1);
  }
}

PointMassTrajectory3D PointMassTrajectory3D::operator=(PointMassTrajectory3D tr){
  return {tr};
}

// Overload for std::vector<Eigen::Matrix<double, 3, 1>>, gates_positions,p_1
std::ostream &operator<<(std::ostream &o, const std::vector<Eigen::Matrix<double, 3, 1>> &vec) {
    o << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        o << vec[i].transpose();  // Transpose for more compact output
        if (i < vec.size() - 1) {
            o << ", ";
        }
    }
    o << "]";
    return o;
}
/*
std::ostream &operator<<(std::ostream &o, const PointMassTrajectory3D &t) {
    o << "pmm3d: t:" << t.time() << "; exists:" << t.exists();
    o << "\n\tx: " << t.x_;
    o << "\n\ty: " << t.y_;
    o << "\n\tz: " << t.z_;
    return o;
}*/
// Overload for PointMassTrajectory3D
std::ostream &operator<<(std::ostream &o, const PointMassTrajectory3D &t) {
  o << "pmm3d: t:" << t.time() << ";exists:" << t.exists();
  o << "\n\tx: " << t.x_;
  o << "\n\ty: " << t.y_;
  o << "\n\tz: " << t.z_;
  return o;
}

} // namespace pmm