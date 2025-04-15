#****** Test Script to test GTSAM - python installation ******#
import gtsam

def test_create_pose2():
  # Create a 2D pose with x, y, and theta (rotation)
  pose = gtsam.Pose2(1.0, 2.0, 0.5)
  print("Pose2 created:", pose)

  return pose

def test_create_prior():
  # Create a prior factor on a Pose2
  prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
  pose_key = gtsam.symbol('x', 1)
  prior_factor = gtsam.PriorFactorPose2(pose_key, gtsam.Pose2(0, 0, 0), prior_noise)
  print("Prior factor created:", prior_factor)
  
  return prior_factor

if __name__ == "__main__":
  # Run basic tests
  pose = test_create_pose2()
  prior = test_create_prior()

  print("GTSAM installation seems to be working!")

