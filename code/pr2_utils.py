
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  # print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))


def show_lidar():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("test_ranges.npy")
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()


def plot_map(mapdata, cmap="binary"):
  plt.imshow(mapdata.T, origin="lower", cmap=cmap)

def test_map():
  # Initialize a grid map
  MAP = {}
  MAP['res'] = np.array([0.05, 0.05])    # meters
  MAP['min'] = np.array([-20.0, -20.0])  # meters
  MAP['max'] = np.array([20.0, 20.0])    # meters
  MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
  isEven = MAP['size']%2==0
  MAP['size'][isEven] = MAP['size'][isEven]+1 # Make sure that the map has an odd size so that the origin is in the center cell
  MAP['map'] = np.zeros(MAP['size'])
  
  # Load Lidar scan
  ranges = np.load("test_ranges.npy")
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  valid1 = np.logical_and((ranges < 30),(ranges> 0.1))
  
  # Lidar points in the sensor frame
  points = np.column_stack((ranges*np.cos(angles), ranges*np.sin(angles)))
  
  # Convert from meters to cells
  cells = np.floor((points - MAP['min']) / MAP['res']).astype(int)
  
  # Insert valid points in the map
  valid2 = np.all((cells >= 0) & (cells < MAP['size']),axis=1)
  MAP['map'][tuple(cells[valid1&valid2].T)] = 1

  # Plot the Lidar points
  fig1 = plt.figure()
  plt.plot(points[:,0],points[:,1],'.k')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Lidar scan')
  plt.axis('equal')
  
  # Plot the grid map
  fig2 = plt.figure()
  plot_map(MAP['map'],cmap='binary')
  plt.title('Grid map')
  
  plt.show()

if __name__ == '__main__':
  show_lidar()
  test_bresenham2D()
  test_map()
  
  
  

