/**
 * @file cftg.cpp
 * @author Krystof Teissing (k.teissing@gmail.com)
 * @version 0.1
 * @date 2023-05-25
 * 
 */

#include "cftg.hpp"

namespace pmm{
std::tuple<double, Vector<3>> getClosestPointOnLineSegment(Vector<3> a, Vector<3> b, Vector<3> p){
  Vector<3> ap = p-a;
  Vector<3> ab = b-a;

  double ab_norm_squared = ab.norm()*ab.norm();
  double ap_dot_ab = ap.dot(ab);
  double dist = ap_dot_ab/ab_norm_squared;

  if (dist < 0){
    return std::tuple(ap.norm(),a);
  } else if (dist > 1) {
    return std::tuple((p-b).norm(),b);
  } else {
    return std::tuple((p-(a + dist*ab)).norm(), (a + dist*ab));
  }
}

std::tuple<bool, Vector<3>> isTrajectoryPathCollisionFree(Scalar drone_radius, std::vector<Vector<3>>* trajectory_path, std::vector<std::vector<Scalar>> map){
    Vector<3> collision_pos;
    for (int wp_idx=0; wp_idx<trajectory_path->size(); wp_idx++){
      bool collision = false; // WARNING: Must be implemented according to application
      if (collision){
        collision_pos = trajectory_path->at(wp_idx);
        return std::tuple(false, collision_pos);
      }
    }
    return std::tuple(true, collision_pos);
}


std::tuple<int, Vector<3>> getClosestPointToPath(std::vector<Vector<3>>* path, Vector<3> p){
    Vector<3> cp;
    Scalar cp_dist = MAXFLOAT;
    int cp_seg_idx = 0;

    for (int seg_idx = 1; seg_idx < path->size(); seg_idx++){
        Scalar dist_tmp;
        Vector<3> cp_tmp;
        std::tie(dist_tmp, cp_tmp) = getClosestPointOnLineSegment(path->at(seg_idx-1), path->at(seg_idx), p);
        if (dist_tmp < cp_dist){
            cp_dist = dist_tmp;
            cp = cp_tmp;
            cp_seg_idx = seg_idx;
        }
    }
    return std::tuple(cp_seg_idx, cp);
}

PMM_MG_Trajectory3D findCollisionFreeTrajectory(std::vector<std::vector<Vector<3>>> path_set, const Scalar collision_distance_check, const double drone_radius, const Scalar max_acc_norm, const Scalar max_vel, std::vector<std::vector<Scalar>> map, bool debug){
  // SELECT THE FASTEST TRAJECTORY AND CHECK FOR COLLISIONS
  // TD parameters
  const int TD_max_iter = 10;
  const Scalar TD_acc_precision = 1e-3;

  // Gradient Method optimization parameters
  // First round
  Scalar alpha = 33.0;
  Scalar alpha_reduction_factor = 0.45;
  Scalar alpha_min_threshold = 1e-3;
  double dT_precision = 1e-3;
  int max_iter = 30;
  // Second round
  Scalar alpha2 = 33.0;
  Scalar alpha_reduction_factor2 = 0.45;
  Scalar alpha_min_threshold2 = 1e-2;
  int max_iter2 = 10;
  bool second_round_opt = true;

  // define heap
  Heap<HeapNode<PMM_MG_Trajectory3D>*> heap;
  // fill heap
  for (int i=0; i<path_set.size(); i++) {
    std::vector<Vector<3>> plan_wpts = path_set[i];

    if (debug){
      std::cout << std::endl;
      std::cout << "Trajectory Id: " << i << std::endl;
      std::cout << "Path waypoints count: " << plan_wpts.size() << std::endl;
    }

    // define trajectory initial conditions
    Vector<3> init_vel = Vector<3>(0.,0.,0.); // start and goal velocity

    // the trajectory computation and optimization is done here
    PMM_MG_Trajectory3D tr_o(plan_wpts, init_vel, init_vel, max_acc_norm, max_vel, dT_precision, max_iter, alpha,
                            alpha_reduction_factor, alpha_min_threshold, TD_max_iter, TD_acc_precision, second_round_opt, 
                            max_iter2, alpha2, alpha_reduction_factor2, alpha_min_threshold2, debug);

    if (tr_o._exists){
      // update the heap2 and trajectory vector
      HeapNode<PMM_MG_Trajectory3D>* traj_node = new HeapNode<PMM_MG_Trajectory3D>();
      traj_node->data.copy_trajectory(tr_o);
      traj_node->distance_from_start = tr_o.duration();
      traj_node->id = i;
      heap.push(traj_node);
    } else {
      std::cout << "WARNING: Skipping path due to infeasible trajectory" << std::endl;
      continue;
    }
  }

  //get the fastest trajectory  
  PMM_MG_Trajectory3D selected_trajectory;
  std::vector<Vector<3>> selected_path;

  bool found_collision_free_trajectory = false;

  PMM_MG_Trajectory3D tr_new;

  while (!found_collision_free_trajectory){
    // get fastest path
    HeapNode<PMM_MG_Trajectory3D>* fastest_traj_node = heap.pop();
    if (fastest_traj_node == NULL){
      return PMM_MG_Trajectory3D();
    }

    selected_path = fastest_traj_node->data.get_current_positions();

    if (debug){
    std::cout << "Fastest traj duration: " << fastest_traj_node->distance_from_start << " , id: " << fastest_traj_node->id << std::endl;
    }

    // COLLISION CHECKING
    // sample trajectory
    std::vector<Scalar> t_sampled;
    std::vector<Vector<3>> traj_path_sampled;
    std::tie(t_sampled, traj_path_sampled) = fastest_traj_node->data.get_sampled_trajectory_path(collision_distance_check);

    // check collision
    bool path_collision_free = true;
    Vector<3> collision_pos;
    std::tie(path_collision_free, collision_pos) = isTrajectoryPathCollisionFree(drone_radius, &selected_path, map);

    if (debug){
    std::cout << "Path is collision free: " << path_collision_free << "    " << collision_pos << std::endl;
    }

    if (!path_collision_free){
      // find closest point from the collision point to the initial path and add it to the path, recompute trajectory
      Vector<3> cp;
      int cp_seg_idx = 0;
      std::tie(cp_seg_idx, cp) = getClosestPointToPath(&selected_path, collision_pos);

      if ((cp-selected_path[cp_seg_idx]).norm() < PRECISION_PMM_VALUES or (cp-selected_path[cp_seg_idx+1]).norm() < PRECISION_PMM_VALUES){
        std::cout << "WARNING: Closest path point to the collision point is already among the path waypoints, removing collision trajectory form heap." << std::endl;
        continue;
      }

      // add the closest point to the path
      selected_path.insert(selected_path.begin() + cp_seg_idx, cp);

      // recompute trajectory
      if (debug){
      std::cout << std::endl;
      std::cout << "Recomputing Trajectory" << std::endl;
      std::cout << "Path waypoints count: " << selected_path.size() << std::endl;
      }

      Vector<3> init_vel = Vector<3>(0.,0.,0.);

      // the trajectory computation and optimization is done here
      tr_new = PMM_MG_Trajectory3D(selected_path, init_vel, init_vel, max_acc_norm, max_vel, dT_precision, max_iter, alpha,
                            alpha_reduction_factor, alpha_min_threshold, TD_max_iter, TD_acc_precision, second_round_opt, 
                            max_iter2, alpha2, alpha_reduction_factor2, alpha_min_threshold2, debug);

      // update the heap and trajectory vector
      if (!tr_new._exists){
        if (debug) {
          std::cout << "WARNING: Trajectory infeasible after adding new waypoint, removing from heap." << std::endl;
        }
        continue;
      } else {
        HeapNode<PMM_MG_Trajectory3D>* traj_node_tmp = new HeapNode<PMM_MG_Trajectory3D>();
        traj_node_tmp->data.copy_trajectory(tr_new);
        traj_node_tmp->distance_from_start = tr_new.duration();
        traj_node_tmp->id = fastest_traj_node->id;
        heap.push(traj_node_tmp);
      }
    } else {
      found_collision_free_trajectory = true;
      selected_trajectory.copy_trajectory(fastest_traj_node->data);
    }
  }
  return selected_trajectory;
}

}