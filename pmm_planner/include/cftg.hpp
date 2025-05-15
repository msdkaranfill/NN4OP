/**
 * @file cftg.hpp
 * @author Krystof Teissing (k.teissing@gmail.com)
 * @version 0.1
 * @date 2023-05-25
 * 
 */

#include "pmm_mg_trajectory3d.hpp"
#include "common.hpp"
#include "heap.hpp"

namespace pmm{
/**
 * @brief Get the Closest Point On Line Segment 
 * 
 * @param a 1st point defining the line
 * @param b 2nd point defining the line
 * @param p point form which the closest point on line AB is computed
 * @return std::tuple<double, Vector<3>> Euclidean distance of p and CP, closest point CP on line AB
 */
std::tuple<double, Vector<3>> getClosestPointOnLineSegment(Vector<3> a, Vector<3> b, Vector<3> p);

/**
 * @brief Get the Closest Point To Path
 * 
 * @param path
 * @param p point form which the closest point on path is computed
 * @return std::tuple<int, Vector<3>> Segment index (stating from 1) where the CP is lying, Closest point CP
 */
std::tuple<int, Vector<3>> getClosestPointToPath(std::vector<Vector<3>>* path, Vector<3> p);

/**
 * @brief Check path for collision, return colliding position if there is a collision
 *        WARNING: Must be implemented according to the application
 * 
 * @param trajectory_path 
 * @return std::tuple<bool, Vector<3>> Flag (false->collision, true->no collision), collision position
 */
std::tuple<bool, Vector<3>> isTrajectoryPathCollisionFree(double drone_radius, std::vector<Vector<3>>* trajectory_path, std::vector<std::vector<Scalar>> map);

/**
 * @brief Find a minimum-time collision-free trajectory
 * 
 * @param paths_set Set of all possible paths
 * @param collision_distance_check Precision for collision checking
 * @param drone_radius Radius of the drone
 * @param max_acc_norm Maximal acceleration norm of the drone given its limited thrust
 * @param map Map in form of Euclidean Signed Distance Field
 * @param debug Flag for console output
 * @return PMM_MG_Trajectory3D 
 */
PMM_MG_Trajectory3D findCollisionFreeTrajectory(std::vector<std::vector<Vector<3>>> paths_set, const Scalar collision_distance_check, const double drone_radius, const Scalar max_acc_norm, const Scalar max_vel, std::vector<std::vector<Scalar>> map, bool debug = false);

}