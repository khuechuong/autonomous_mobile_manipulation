#!/usr/bin/env python
import rospy
import numpy as np
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
import utils as u
from scipy.spatial.transform import Rotation as R
from tf.transformations import *
# import tf2_ros
# import tf2_geometry_msgs
# import tf_conversions

DMIN = 0.3  # [m]
DMAX = 0.85  # [m]

class FullCoverage:
    def __init__(self):
        rospy.init_node('full_coverage', anonymous=True)

        # move_base 2d navigation
        self.client = SimpleActionClient('bvr_SIM/move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()

        """publishers for rviz & arm"""
        self.publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.arm_goal = rospy.Publisher('arm_goal', Marker, queue_size=10)
        self.enddefector_goal_pose_pub_ = rospy.Publisher("/endeffector_goal_pose",PoseStamped, queue_size=10)
        self.joy_pub = rospy.Publisher('rviz_visual_tools_gui', Joy, queue_size=10)
        
        self.success_sub_ = rospy.Subscriber("/bvr_SIM/success", Bool, self.successCb)
        self.success_msg = None

        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.plot_points)
        """ 2d goal index   """
        self.goal_index = 0
        
        # rospy.sleep(1)  # Wait to establish connection to RViz
        self.point_reached = []
        self.point_failed = []
        self.point_failed_again = []
        self.path = None
        csv_path = "/home/ara1804/autonomous_mobile_manipulation_ws/src/autonomous_mobile_manipulation/gazebo_resources/model_facets/boat.csv"
        """ load coverage points    """
        array_from_csv = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        """ generate viewpoints """
        viewpoints = u.calculate_viewpoints(array_from_csv, 0.3, z_min=0.3)
        """ generate a boundary offset of 0.4   """
        self.bound = u.create_boundary(array_from_csv, 0.8, False, 0.3)
        """ divide viewpoints into clusters and cluster base locations """
        l1, l2 = u.slice_3d_viewpoints(viewpoints, self.bound, DMIN, DMAX, array_from_csv[:, :3], False, 7)
        origin = np.array([0, 0]) 
        # config base_location into 2d
        self.base_location_2d = np.array(l2)[:, :2]
        # get index order of best path
        points = np.vstack([origin, self.base_location_2d])         
        self.tour = u.nearest_neighbor_tsp(points, start_index=0, plot=False, model=array_from_csv[:,:2])
        
        """ format 3d path  """
        array_2d = np.vstack(l2)
        """ Number of rows in the array """
        num_rows = array_2d.shape[0]
        """Pad the array with zeros to have 7 columns"""
        padded_array = np.pad(array_2d, ((0, 0), (0, 4)), 'constant', constant_values=0)
        """ get path backwards so I can pop easily  """
        self.path = u.plot_tsp_paths(l1,padded_array, False)
        
        """ pre-make Joy    """
        self.joy_msg = Joy()
        self.joy_msg.header.seq = 0
        self.joy_msg.header.stamp = rospy.Time.now()
        self.joy_msg.header.frame_id = "map"
        self.joy_msg.buttons = [0,1,0,0,0,0]        

    def send_goals(self):
        for i in range(1,len(self.tour)-1):
            # -1 b/c first goal in tour starts at 1
            self.goal_index = self.tour[i] - 1
            base_loc_x = self.base_location_2d[self.goal_index, 0]
            base_loc_y = self.base_location_2d[self.goal_index, 1]

            cluster = self.path[self.goal_index]
            if cluster.shape[0] > 0:
                """ base location pose orientation  """
                avg_x = np.mean(cluster[:,0])
                avg_y = np.mean(cluster[:,1])
                angle = np.arctan2(avg_y-base_loc_y, avg_x-base_loc_x)
                r = R.from_euler('z', angle, degrees=False)
                quaternion = r.as_quat()

                """ go to cluster base location """
                self.send_2d_goal(base_loc_x, base_loc_y, quaternion[0],quaternion[1],quaternion[2],quaternion[3])
                """ inspect cluster points """                
                for point_index in range(len(cluster)-1, -1, -1):
                    goal_point = cluster[point_index]
                    quat = goal_point[3:]
                    quat_tf = quaternion_from_euler(np.deg2rad(180), np.deg2rad(0), np.deg2rad(180))
                    new_quat = quaternion_multiply(quat, quat_tf)
                    """ inspect cluster, orientation negative """
                    # successful = self.send_3d_goal(goal_point[0],goal_point[1],goal_point[2],goal_point[3],goal_point[4],goal_point[5],goal_point[6])
                    successful = self.send_3d_goal(goal_point[0],goal_point[1],goal_point[2],new_quat[0],new_quat[1],new_quat[2],new_quat[3])
                    """ delete whether success or fail  """
                    self.path[0] = self.path[0][:-1]

                if len(self.point_failed) > 0:
                    rospy.logwarn("Redoing the failed points")
                    stacked = np.vstack(self.point_failed)
                    avg_x = np.mean(stacked[:, 0])
                    avg_y = np.mean(stacked[:, 1])
                    distances = np.linalg.norm(self.bound[:, :2] - np.array([avg_x, avg_y]), axis=1)
                    closest_index = np.argmin(distances)
                    """ new position    """
                    closest_point = self.bound[closest_index]
                    angle = np.arctan2(avg_y-base_loc_y, avg_x-base_loc_x)
                    r = R.from_euler('z', angle, degrees=False)
                    """ new orientation """
                    quaternion = r.as_quat()
                    """ go to new base location """
                    self.send_2d_goal(closest_point[0], closest_point[1], quaternion[0],quaternion[1],quaternion[2],quaternion[3])
                    """ inspect failed points   """
                    for point_index in range(len(self.point_failed)):
                        goal_point = self.point_failed[-1]
                        """ inspect cluster, orientation negative """
                        successful = self.send_3d_goal(goal_point[0],goal_point[1],goal_point[2],goal_point[3],goal_point[4],goal_point[5],goal_point[6], True)
                        """ delete whether success or fail  """
                        self.point_failed = self.point_failed[:-1]


    
    def test(self):
        """ go to cluster base location """
        self.send_2d_goal(0.75,0)

        """ inspect cluster """
        successful = self.send_3d_goal(1.75,0,0.75,0,0.383,0,0.924)
        print(successful)

        """ go to cluster base location """
        self.send_2d_goal(0,0)

        """ inspect cluster """
        successful = self.send_3d_goal(1.75,0,0.75,0,0.383,0,1)
        print(successful)
        # TODO: moveit
        # if rospy.is_shutdown():
        #     break    

    def send_3d_goal(self,p_x, p_y, p_z, o_x=0.0, o_y=0.0, o_z=0.0, o_w=1.0, second_time=False):
        """ Send goal to endefector 
            Return if successful        """
        rate_pose = rospy.Rate(10)
        start_time = rospy.Time.now()
        publish_log = False
        while rospy.Time.now() - start_time < rospy.Duration(1):
            if not publish_log:
                rospy.logwarn("Publishing PoseStamped msg for 1 secs")
                publish_log = True
            self.publish_pose(p_x, p_y, p_z, o_x, o_y, o_z, o_w)
            self.plot_3d_goal(p_x, p_y, p_z, o_x, o_y, o_z, o_w)
            rate_pose.sleep()

        rospy.loginfo("Publishing Joy once")
        self.joy_pub.publish(self.joy_msg)

        success_log = False
        while self.success_msg is None:
            if not publish_log:
                rospy.loginfo("success unknown")
                success_log = True
        
        rospy.loginfo("Success: %s", self.success_msg)
        
        if self.success_msg:
            rospy.loginfo("Waiting moveit for 12 secs")
            rospy.sleep(12)
            rospy.loginfo("Moveit Done!")
            self.point_reached.append(np.array([p_x, p_y, p_z, o_x, o_y, o_z, o_w]))
        else:
            """ add to self.point_failed_again if it's second time """
            if second_time:
                rospy.logerr("Planning failed, add to failed again, move on")
                self.point_failed_again.append(np.array([p_x, p_y, p_z, o_x, o_y, o_z, o_w]))
            else:
                rospy.logerr("Planning failed, add to failed, move on")
                self.point_failed.append(np.array([p_x, p_y, p_z, o_x, o_y, o_z, o_w]))
        
        success_ = self.success_msg
        """ set None for next   """
        self.success_msg = None    
        return success_

    def publish_pose(self,p_x, p_y, p_z, o_x=0.0, o_y=0.0, o_z=0.0, o_w=1.0):
        """ A simple PoseStamped publisher  """
        pose_msg = PoseStamped()
        pose_msg.header.seq = 0
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = p_x
        pose_msg.pose.position.y = p_y
        pose_msg.pose.position.z = p_z

        pose_msg.pose.orientation.x = o_x
        pose_msg.pose.orientation.y = o_y
        pose_msg.pose.orientation.z = o_z
        pose_msg.pose.orientation.w = o_w

        self.enddefector_goal_pose_pub_.publish(pose_msg)      
    
    def send_2d_goal(self,x,y,q_x = 0, q_y=0, q_z=0, q_w=1):
        """ROS Navigation stack move_base to send 2d goal   """
        move_goal = MoveBaseGoal()
        move_goal.target_pose.header.frame_id = "map"
        move_goal.target_pose.header.stamp = rospy.Time.now()
        # Setting the position
        move_goal.target_pose.pose.position.x = x
        move_goal.target_pose.pose.position.y = y
        move_goal.target_pose.pose.orientation.x = q_x 
        move_goal.target_pose.pose.orientation.y = q_y 
        move_goal.target_pose.pose.orientation.z = q_z 
        move_goal.target_pose.pose.orientation.w = q_w 

        rospy.loginfo("Sending goal to move_base")
        self.client.send_goal(move_goal)
        self.client.wait_for_result()
        rospy.loginfo("Goal reached.")

    def plot_points(self, event=None):
        if self.path is None:
            return
        """ Plots points to RViz, highlighting the points from the specified index in yellow. """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Point width
        marker.scale.y = 0.02  # Point height

        # Set marker colors and points
        for i, points in enumerate(self.path):
            for point in points:  # each row is a point
                p = Point()
                p.x, p.y, p.z = point[:3]  # extract x, y, z from the first three columns
                marker.points.append(p)
                
                color = ColorRGBA()
                color.a = 1.0  # Alpha must be set to 1 for the color to be opaque
                if i == self.goal_index:
                    color.r, color.g, color.b = 1.0, 1.0, 0.0  # Yellow
                else:
                    color.r, color.g, color.b = 0.0, 0.0, 1.0  # Blue
                marker.colors.append(color)

        for point in self.point_reached:
            p = Point()
            p.x, p.y, p.z = point[:3]  # extract x, y, z from the first three columns
            marker.points.append(p)
            
            color = ColorRGBA()
            color.a = 1.0
            color.r, color.g, color.b = 0.0, 1.0, 0.0
            marker.colors.append(color)
        
        for point in self.point_failed:
            p = Point()
            p.x, p.y, p.z = point[:3]  # extract x, y, z from the first three columns
            marker.points.append(p)
            
            color = ColorRGBA()
            color.a = 1.0
            color.r, color.g, color.b = 1.0, 0.0, 0.0
            marker.colors.append(color)

        for point in self.point_failed_again:
            p = Point()
            p.x, p.y, p.z = point[:3]  # extract x, y, z from the first three columns
            marker.points.append(p)
            
            color = ColorRGBA()
            color.a = 1.0
            color.r, color.g, color.b = 1.0, 0.0, 1.0
            marker.colors.append(color)

        self.publisher.publish(marker)

    def plot_3d_goal(self,p_x, p_y, p_z, o_x=0.0, o_y=0.0, o_z=0.0, o_w=1.0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "arrows"
        marker.id = 1000
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.11  # Shaft diameter
        marker.scale.y = 0.03  # Head diameter
        marker.scale.z = 0.02  # Head length
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Position (x, y, z)
        marker.pose.position.x = p_x
        marker.pose.position.y = p_y
        marker.pose.position.z = p_z

        # Orientation (x, y, z, w)
        marker.pose.orientation.x = o_x
        marker.pose.orientation.y = o_y
        marker.pose.orientation.z = o_z
        marker.pose.orientation.w = o_w

        self.arm_goal.publish(marker)
    

    def successCb(self, msg):
        self.success_msg = msg.data

if __name__ == '__main__':
    goal_sender = FullCoverage()
    try:
        goal_sender.send_goals()
        # goal_sender.test()
    except rospy.ROSInterruptException:
        pass




# TODO: choose 2 closest points of each cluster
# TODO: write a code moveit


