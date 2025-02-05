import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
from typing import List, Dict, Tuple

class PhysicalObject:
    def __init__(self):
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.affected_by_agents = False
    
    def update(self, agents: List['Agent'], dt: float):
        pass

class Seesaw(PhysicalObject):
    def __init__(self, center_position: np.ndarray, length: float = 6.0):
        super().__init__()
        self.position = np.array(center_position)
        self.length = length
        self.angle = 0  # Rotation around y-axis
        self.affected_by_agents = True
        self.angular_velocity = 0
        self.damping = 2.0
        self.stiffness = 5.0
        
    def get_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        dx = self.length/2 * np.cos(self.angle)
        dz = self.length/2 * np.sin(self.angle)
        left_end = self.position + np.array([-dx, 0, -dz])
        right_end = self.position + np.array([dx, 0, dz])
        return left_end, right_end
    
    def update(self, agents: List['Agent'], dt: float):
        # Calculate torque based on agent positions
        total_torque = 0
        left_end, right_end = self.get_endpoints()
        
        for agent in agents:
            # Check if agent is close enough to affect the seesaw
            dist_to_left = np.linalg.norm(agent.position - left_end)
            dist_to_right = np.linalg.norm(agent.position - right_end)
            
            if dist_to_left < 1.0:  # Agent is on left side
                total_torque -= 1.0
            elif dist_to_right < 1.0:  # Agent is on right side
                total_torque += 1.0
        
        # Add spring force to return to horizontal
        spring_torque = -self.stiffness * self.angle
        
        # Add damping
        damping_torque = -self.damping * self.angular_velocity
        
        # Update angular velocity and angle
        self.angular_velocity += (total_torque + spring_torque + damping_torque) * dt
        self.angle += self.angular_velocity * dt
        
        # Limit angle
        self.angle = np.clip(self.angle, -np.pi/4, np.pi/4)

class Agent:
    def __init__(self, id: int, start: np.ndarray, goal: np.ndarray, color: str):
        self.id = id
        self.position = np.array(start, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.color = color
        self.radius = 0.3
        self.max_speed = 0.5
        self.path = []
        self.reached_goal = False
        
    def update(self, objects: List[PhysicalObject], other_agents: List['Agent'], dt: float):
        if self.reached_goal:
            return
            
        # Calculate direction to goal
        to_goal = self.goal - self.position
        distance_to_goal = np.linalg.norm(to_goal)
        
        if distance_to_goal < 0.1:
            self.reached_goal = True
            return
            
        # Basic movement towards goal
        desired_velocity = (to_goal / distance_to_goal) * self.max_speed
        
        # Avoid other agents
        avoid_force = np.zeros(3)
        for other in other_agents:
            if other.id != self.id:
                diff = self.position - other.position
                dist = np.linalg.norm(diff)
                if dist < 1.0:
                    avoid_force += diff / (dist ** 2)
        
        # Update velocity with avoidance
        self.velocity = desired_velocity + avoid_force * 0.5
        
        # Check for physical interactions
        for obj in objects:
            if isinstance(obj, Seesaw):
                left_end, right_end = obj.get_endpoints()
                # If agent is on the seesaw, adjust height accordingly
                if np.linalg.norm(self.position[:2] - left_end[:2]) < 1.0:
                    self.position[2] = left_end[2]
                elif np.linalg.norm(self.position[:2] - right_end[:2]) < 1.0:
                    self.position[2] = right_end[2]
        
        # Update position
        self.position += self.velocity * dt
        self.path.append(self.position.copy())

class Environment:
    def __init__(self, bounds: List[List[float]]):
        self.bounds = bounds
        self.agents: List[Agent] = []
        self.physical_objects: List[PhysicalObject] = []
        
    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        
    def add_physical_object(self, obj: PhysicalObject):
        self.physical_objects.append(obj)
        
    def update(self, dt: float):
        # Update physical objects
        for obj in self.physical_objects:
            obj.update(self.agents, dt)
        
        # Update agents
        for agent in self.agents:
            agent.update(self.physical_objects, 
                        [a for a in self.agents if a.id != agent.id], 
                        dt)

def create_visualization():
    # Create environment
    bounds = [[0, 10], [0, 10], [0, 10]]
    env = Environment(bounds)
    
    # Add seesaw
    seesaw = Seesaw([5, 5, 3])
    env.add_physical_object(seesaw)
    
    # Add agents
    agents = [
        Agent(0, np.array([2, 5, 3]), np.array([8, 5, 3]), 'red'),
        Agent(1, np.array([8, 3, 3]), np.array([2, 7, 3]), 'blue'),
        Agent(2, np.array([2, 7, 3]), np.array([8, 3, 3]), 'green')
    ]
    for agent in agents:
        env.add_agent(agent)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        
        # Update environment
        env.update(0.1)
        
        # Set plot limits
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Draw seesaw
        left_end, right_end = seesaw.get_endpoints()
        ax.plot([left_end[0], right_end[0]], 
                [left_end[1], right_end[1]], 
                [left_end[2], right_end[2]], 
                'k-', linewidth=3)
        
        # Draw center support
        ax.plot([seesaw.position[0], seesaw.position[0]], 
                [seesaw.position[1], seesaw.position[1]], 
                [0, seesaw.position[2]], 
                'k-', linewidth=2)
        
        # Draw agents
        for agent in agents:
            ax.scatter(*agent.position, color=agent.color, s=100)
            if len(agent.path) > 1:
                path = np.array(agent.path)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                       color=agent.color, alpha=0.3)
        
        ax.set_title(f'Multi-Agent Path Planning with Physics - Frame {frame}')
    
    anim = FuncAnimation(fig, update, frames=200, interval=50, repeat=True)
    plt.show()

if __name__ == "__main__":
    create_visualization()