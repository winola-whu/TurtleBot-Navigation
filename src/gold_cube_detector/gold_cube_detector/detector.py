import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool

class GoldCubeDetector(Node):
    def __init__(self):
        super().__init__('gold_cube_detector_with_cloud_match')
        self.bridge = CvBridge()
        # Subscribe to RTAB-Map topics
        self.image_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/depth', self.depth_callback, 10)
        self.cloud_sub = self.create_subscription(
            PointCloud2, '/rtabmap/cloud_map', self.cloud_callback, 1)
        self.goal_pub = self.create_publisher(Bool, "/goal_reached", 10)
        self.latest_depth = None
        self.latest_cloud = None
        self.fx = 277.0
        self.fy = 277.0
        self.cx = 160.0
        self.cy = 120.0
        self.detected_count = 0
        self.required_consistent = 15
        self.goal_sent = False

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"CV bridge depth conversion failed: {e}")

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    def image_callback(self, img_msg):
        self.get_logger().info("Image received")
        try:
            color_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV bridge conversion failed: {e}")
            return

        if self.latest_depth is None:
            self.get_logger().warn("No depth image received yet.")
            return

        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        lower_gold = np.array([19, 134, 196])  # Tune as needed for your cube
        upper_gold = np.array([39, 154, 216])
        mask = cv2.inRange(hsv, lower_gold, upper_gold)

        cv2.imshow("Gold Mask", mask)
        cv2.waitKey(1)

        unique = np.unique(mask)
        self.get_logger().info(f"Mask unique values: {unique}")
        if np.count_nonzero(mask) == 0:
            self.get_logger().warn("No gold-colored region detected in this frame.")
            self.detected_count = 0
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().warn("No contours found in mask.")
            self.detected_count = 0
            return

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 50:
            self.get_logger().warn("Largest region too small, likely not the cube.")
            self.detected_count = 0
            return

        x, y, w, h = cv2.boundingRect(contour)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        self.get_logger().info(f"Cube detected at pixel: ({cx}, {cy})")
        self.detected_count += 1

        depth = float(self.latest_depth[cy, cx])
        if np.isnan(depth) or depth == 0.0:
            self.get_logger().warn("No valid depth at centroid.")
            self.detected_count = 0
            return

        if self.detected_count >= self.required_consistent and not self.goal_sent:
            self.goal_sent = True
            self.goal_pub.publish(Bool(data=True))
            self.get_logger().info("Published goal reached message!")

        # 3D position in camera frame
        X = (cx - self.cx) * depth / self.fx
        Y = (cy - self.cy) * depth / self.fy
        Z = depth
        self.get_logger().info(f"Gold cube position in camera frame: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m")

        # Search for closest point in the global point cloud
        if self.latest_cloud is not None:
            min_dist = float('inf')
            closest_pt = None
            for pt in point_cloud2.read_points(self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True):
                dist = math.sqrt((pt[0]-X)**2 + (pt[1]-Y)**2 + (pt[2]-Z)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_pt = pt
            if closest_pt is not None:
                self.get_logger().info(f"Closest global cloud point: x={closest_pt[0]:.2f}, y={closest_pt[1]:.2f}, z={closest_pt[2]:.2f} (distance: {min_dist:.3f} m)")
            else:
                self.get_logger().warn("No valid point found in global cloud.")
        else:
            self.get_logger().warn("No point cloud received yet.")

def main(args=None):
    rclpy.init(args=args)
    node = GoldCubeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
