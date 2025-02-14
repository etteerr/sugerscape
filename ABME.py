import numpy as np
import matplotlib.pylab as pl
from IPython import display
import random
import time
from numba import jit
#from IPython.core.debugger import set_trace
import sys
import copy
from matplotlib import animation, rc
from IPython.display import HTML


class Grid():
    neighbourhood_dict = {
        'left': np.array([-1, 0]),
        'right': np.array([1, 0]),
        'up':   np.array([0, 1]),
        'down': np.array([0, -1])
    }
    resources = dict()

    def __init__(self, size):
        self.agents = np.zeros([size, size])
        self.width = size
        self.height = size
        self.size = size
        self.resources = dict()

        #'Legacy' variables
        self.neighbourhood = np.array(
            [val for key, val in self.neighbourhood_dict.items()])
        self.neighbourhood_names = np.array(
            [key for key, val in self.neighbourhood_dict.items()])

        # Assert vectors are alligned
        for key, val in self.neighbourhood_dict.items():
            assert(val == self.neighbourhood[
                   self.neighbourhood_names == key]).all()

    '''
        Initializes the agents position by placing the agent in the grid
        on a random position or on [position] if [position] is not None
    '''

    def place_agent(self, agent, position=None):
        agents = self.agents
        prevpos = agent.position
        if position is None:
            width = self.width
            height = self.height
            assert(width == height)  # Assumption width == height
            while True:
                position = np.random.randint(0, width, size=2)
                if agents[position[0]][position[1]] == 0:
                    break

        if agents[position[0]][position[1]] > 0:
            print("Failed to place agent: Position already occupied")
            return -1
        agent.position = position

        # update map (remove old)
        if prevpos is not None:
            if (agents[prevpos[0]][prevpos[1]]) == agent.id:
                self.agents[prevpos[0]][prevpos[1]] = 0
        # update map (add new)
        self.agents[position[0]][position[1]] = agent.id
        agent.grid = self
        return position

    '''
        returns a list of (absolute) neigbhourhood coordinates based on the
        neighbourhood as specified by [self.neighbourhood]
    '''

    def get_neighbourhood_coordinates(self, agent):
        pos = np.array(agent.position)
        size = self.size
        neighbourhood = self.neighbourhood
        neigh = np.array([(pos + delta) % size for delta in neighbourhood])
        return neigh

    '''
        Same as get_neighbourhood_coordinates
        But does not return occupied coordinates.
    '''

    def get_move_canidates(self, agent):
        cpos = agent.position
        size = self.size
        neighbourhood = self.neighbourhood
        agents = self.agents
        assert(self.agents[cpos[0]][cpos[1]] == agent.id)
        canidates = []
        for delta in neighbourhood:
            pos = (cpos + delta) % size
            if agents[pos[0]][pos[1]] == 0:
                canidates.append(pos)
        return canidates

    '''
        Moves agent by:
            - move_name as a name from neighbourhood_names ('left', 'right', etc)
            - move_to as a exact coordinate ([32,11], etc)
            - move as a delta value ([0,1], [1,0] etc)
            - move_idx as the index of the move in neighbourhood
            - None specified is random uniform movement
        Note that every move is valid as long as it does not intrude on a occupied position,
        in which case this function returns False
    '''

    def move_agent(self, agent, move_name=None, move_to=None, move=None, move_idx=None):
        cpos = agent.position
        assert(self.agents[cpos[0]][cpos[1]] == agent.id), \
            'Agent is not present at this position! (Only change agent.position through Grid)'

        npos = move_to

        if move_name is not None:
            npos = cpos + self.neighbourhood_dict[move_name]

        if move is not None:
            npos = cpos + move

        if move_idx is not None:
            npos = cpos + self.neighbourhood[move_idx]

        if npos is None:
            canidates = self.get_move_canidates(agent)
            if len(canidates) > 0:
                npos = random.choice(canidates)
            else:
                return False

        npos = np.array(npos)

        npos %= self.size

        if self.agents[npos[0]][npos[1]] > 0:
            return False

        self.agents[cpos[0]][cpos[1]] = 0
        self.agents[npos[0]][npos[1]] = agent.id

        agent.position = np.array(npos)
        return True

    '''
        Shows a figure of the location of all the agents (can be called in a loop)
    '''

    def image_agents(self):
        pl.figure('grid_agents', figsize=(10, 10))
        pl.imshow(self.agents, interpolation='nearest',
                  aspect='auto', origin='lower')

    def resource_add(self, resource):
        name = resource.name
        for n in self.resources:
            if name == n:
                print("Resource %s already in the grid" % name)
                return False
        self.resources[name] = resource
        return True

    def resource_available(self):
        return [r.name for r in self.resources]

    def remove_agent(self, agent):
        self.agents[self.agents == agent.id] = 0

    def harvest(self, agent, efficiency):
        for key, resource in self.resources.items():
            resource.harvest(agent, efficiency[key])


class Agent():
    # no touchy
    position = None
    id = None
    grid = None
    model = None

    '''
        Describe the agent as a string
    '''

    def __str__(self):
        description = []
        description.append("Agent %i" % self.id)

        fin_str = ""
        for i in description:
            fin_str += i + "\n"
        return fin_str

    '''
        Initialize agent specific variables here
        This function is called when agent is added to model (model.add_agent(...,init_agent=True))
    '''

    def init(self):
        assert(self.model is not None), "Model is None, was agent added to model?"
        assert(self.grid is not None), "Grid is None, Agent was not placed in a grid. This agent requires a grid"

    '''
        Define how the agent moves
    '''

    def move(self):
        print("agent %i moves" % self.id)

    '''
        Define what the agent does (except for movement)
    '''

    def act(self):
        print("agent %i acts" % self.id)

    '''
        Looks around
        returns:
            dictionary with resource names containing:
                Dictionaries with neighbourhood names
                    vector of resources seen with the length equal to self.vision
                Also:
                    ['current'] for each resource
        in Grid can be found:
            neighbourhood = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]);
            neighbourhood_names = ['left', 'right', 'up', 'down'];
    '''

    def look(self):
        assert(self.grid is not None), 'Grid is required to look'
        assert(len(self.grid.resources)
               ), "Resources required for the look function"
        # size
        size = self.grid.size
        # Get current position
        cx = self.position[0]
        cy = self.position[1]

        # Neighbourhood information
        neighbourhood = self.grid.neighbourhood
        neighbourhood_names = self.grid.neighbourhood_names

        # Gather resource data
        r_vision = dict()
        for r_name, resource in self.grid.resources.items():
            r_vision[r_name] = dict()
            # Center
            r_vision[r_name]['current'] = resource.resourceMap[cx][cy]
            # Surround
            for i in range(len(neighbourhood_names)):
                n_name = neighbourhood_names[i]
                n_vect = neighbourhood[i]
                # Rewrite to vector operation
                r_vision[r_name][n_name] = \
                    [resource.resourceMap[(cx + (dx * n_vect[0])) % size][(cy + (dx * n_vect[1])) % size]
                     for dx in range(1, self.vision + 1)]
        return r_vision


class Model():
    tick = 0
    num_agents = 0
    grid = None
    agents = []
    id_tick = 1
    step_order = 'random'
    trackers = []
    dead = [];

    def __init__(self):
        self.tick = 0
        self.num_agents = 0
        self.grid = None
        self.agents = []
        self.id_tick = 1
        self.step_order = 'random'

    def set_grid(self, grid):
        self.grid = grid

    '''
        Add a agent to the model (and assigns ID)
        if place_in_grid, it will also add the agent to the current grid
        If position is specified, it will try to place the agent at the specified position if place_in_grid is true
        
        Returns True on success
    '''

    def add_agent(self, agent, place_in_grid=True, position=None, init_agent=True):
        assert(not (place_in_grid and self.grid is None)
               ), 'Place in grid specified while there is no grid atatched to the model'

        if (agent.id is None):
            agent.id = self.id_tick
            self.id_tick += 1

        if place_in_grid:
            pos = self.grid.place_agent(agent, position)
            if list(pos) == -1:
                return False

        self.agents.append(agent)

        agent.model = self

        if init_agent:
            agent.init()

        return True

    def remove_agent(self, agent):
        if self.grid is not None:
            self.grid.remove_agent(agent)
        self.agents.remove(agent)

    def step_move(self):
        step_order = [i for i in self.agents]
        if self.step_order == 'random':
            random.shuffle(step_order)

        for agent in step_order:
            agent.move()

    def step_act(self):
        step_order = [i for i in self.agents]
        if self.step_order == 'random':
            random.shuffle(step_order)

        for agent in step_order:
            agent.act()
        
        self.tick += 1;

    def step(self):
        self.step_move()
        self.step_act()
        self.track()

    def track(self):
        if self.trackers is not None:
            for tracker in self.trackers:
                tracker.track()

    def recombination(self, agent1, agent2):
        new_agent = SugerAgent()
        agent1.genome

        for i in range(len(new_agent.genome)):
            new_agent.genome[i] = int(
                (agent1.genome[i] + agent2.genome[i]) / 2)

        new_agent.place_agent()

    def next_population(self):
        agents = copy.copy(self.agents)
        agents.sort()

        best10procent = int(self.num_agents * 0.1)
        num_offspring = int(self.num_agents / best10procent)
        extra = self.num_agents - best10procent * num_offspring
        for i in range(best10procent):
            for j in range(num_offspring + (i < extra)):
                recombination(agent[i], random.choice(agents))

        for agent in agents:
            agent.die()


class Resource():
    resourceMap = None
    size = None
    amount = None  # Used when no grid is specified
    capacity = None
    name = 'Unset'

    def __init__(self, name, grid):
        self.resourceMap = np.zeros([grid.size, grid.size])
        self.size = grid.size
        self.name = name
        grid.resource_add(self)

    '''
        Adds resources from a single point source.
        Capacity is always applied if set
            example funs:
                static growth:
                    - lambda x: 1
                circular growth with strength 4 (integers)
                    - lambda x: np.round(4/((0.1*x)+1))
    '''

    def grow(self, cx, cy, fun=lambda x: np.round(4 / ((0.1 * x) + 1)), radius=None, invert=False, apply=True, scale=True):
        field = [[np.sqrt((x - cx)**2 + (y - cy)**2)
                  for y in range(self.size)] for x in range(self.size)]
        field = np.array(field)

        if radius is not None:
            mask = field > radius
            if invert:
                m = np.max(field)
                field = m / (field + 1)
            if scale:
                field /= np.sqrt(2 * (self.size**2))
            field = fun(field)
            field[mask] = 0
        else:
            if invert:
                m = np.max(field)
                field = m / (field + 1)
            if scale:
                field /= np.sqrt(2 * (self.size**2))
            field = fun(field)
        # Add field of resources to resourceMap;
        self.resourceMap += field

        if apply:
            self.apply_capacity()

    '''
        Decay (or apply function to field)
            Applies the function [fun] to all elements of the field
            Default function decays each cell by 5%
            
        Note that this function can also grow resources :)
    '''

    def decay(self, fun=lambda x: 0.95 * x):
        self.resourceMap = fun(self.resourceMap)

    def set_capacity(self, cx, cy, fun=lambda x: np.round(4 / ((0.1 * x) + 1)), radius=None, invert=False, apply=True, scale=True):
        field = [[np.sqrt((x - cx)**2 + (y - cy)**2)
                  for y in range(self.size)] for x in range(self.size)]
        field = np.array(field)

        if radius is not None:
            mask = field > radius
            if invert:
                m = np.max(field)
                field = m / (field + 1)
            if scale:
                field /= np.sqrt(2 * (self.size**2))
            field = fun(field)
            field[mask] = 0
        else:
            if invert:
                m = np.max(field)
                field = m / (field + 1)
            if scale:
                field /= np.sqrt(2 * (self.size**2))
            field = fun(field)
        # Set capacity
        if self.capacity is None:
            self.capacity = np.zeros([self.size, self.size])
        self.capacity[field > self.capacity] = field[field > self.capacity]
        # Apply capacity
        if apply:
            self.apply_capacity()

    def remove_capacity(self):
        self.capacity = None

    def apply_capacity(self):
        if self.capacity is not None:
            r = self.resourceMap
            c = self.capacity
            mask = r > c
            r[mask] = c[mask]
            self.resourceMap = r

    def harvest(self, agent, efficiency=1):
        (x, y) = agent.position
        agent.wealth[self.name] += self.resourceMap[x][y] * efficiency
        self.resourceMap[x][y] = 0


class Visualizer():
    import matplotlib.pyplot as pl
    '''
        Visualizer class.
        Takes a model class as initializer
        Helps with visualizing things
    '''

    def __init__(self, Model):
        '''Visualizer init'''
        self.model = Model

    def clear(self):
        display.clear_output(wait=True)

    def show(self, fig=None):
        if fig is None:
            display.display();
        else:
            display.display(fig)

    def plot_grid(self, return_axis=True, return_fig=False, new_fig=True, legend=False, scale_to_capacity=True):
        '''
            Plots resources (max 2) and agents in a image plot
            using the 3 color channels

            Required:
                model.grid
                model.grid.resources (length == 2)
        '''
        assert(self.model.grid is not None), "grid required to plot grid"
        assert(len(self.model.grid.resources) <= 2), "Requires 2 resources"

        # Init figure
        if new_fig:
            fig = pl.figure()
        else:
            fig = pl.gcf()  # Get current fig

        # Gather data
        r_names = list(self.model.grid.resources.keys())
        # Gather r1 & r2
        r1 = self.model.grid.resources[r_names[0]]
        r2 = self.model.grid.resources[r_names[1]]

        # Create resource mapping
        if scale_to_capacity:
            r = r1.resourceMap / np.max(r1.capacity)
            g = r2.resourceMap / np.max(r2.capacity)
        else:
            r = r1.resourceMap / np.max(r1.resourceMap)
            g = r2.resourceMap / np.max(r2.resourceMap)

        # Map agents
        b = copy.copy(self.model.grid.agents)
        b[b > 0] = 1.0

        # Create image (by stacking channels)
        data = np.dstack((r, b, g))

        # Plot image (img contains AxesImage)
        img = pl.imshow(np.transpose(data, axes=(1, 0, 2)),
                        interpolation='nearest', origin='down')

        # Return request
        if return_axis:
            return img
        if return_fig:
            return fig
        return None  # default if no return is specified

    def animate_axis_array(self, array, fig, html_vid=True, filename=None, fps=15, clear_output=False):
        if clear_output:
            self.clear()

        ani = animation.ArtistAnimation(fig, array, interval=np.ceil(
            1000 / fps), blit=True, repeat_delay=0)

        if html_vid:
            hf = HTML(ani.to_html5_video())
            self.show(hf)

        if filename is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            ani.save('sim.mp4', writer=writer)

    def plot_wealth_hist(self, show=True, return_fig=False):
        """
        Create histogram of all resources defined

        Returns list of list of figure and axis if return_fig is true
        """
        agents = self.model.agents
        resources = self.model.grid.resources
        pl.ioff()  # suppress plotting before show()
        if return_fig:
            figs = []
        for resource in resources:
            fig, ax = pl.subplots()
            ax.hist([agent.wealth[resource] for agent in agents])
            ax.set_title("Wealth histogram for resource %s" % (resource))
            ax.set_xlabel("Wealth")
            ax.set_ylabel("Count")
            if show:
                fig.show()
            if return_fig:
                figs.append([fig, ax])  # also add ax to modify figure
        pl.ion()  # re-enable interactive plotting
        if return_fig:
            return figs

    def plot_needs_metabolism(self, show=True, return_fig=False, seperate_above_mean=True):
        """
        Create scatterplot with harvest efficiency of resource 1 times metabolism on x-axis
        and resource 2 on y

        Returns figure and axis if return_fig is true        
        """

        agents = self.model.agents
        resources = list(self.model.grid.resources)
        if seperate_above_mean:
            average_fitness = np.mean([a.fitness() for a in agents]);
        else:
            average_fitness = -9999;
            
        pl.ioff()  # suppress plotting before show()
        fig, ax = pl.subplots()
        
        if seperate_above_mean: 
            x = [agent.needs[resources[0]] *
                 agent.genome[0] for agent in agents if agent.fitness() < average_fitness]
            y = [agent.needs[resources[1]] *
                 agent.genome[0] for agent in agents if agent.fitness() < average_fitness]
            ax.scatter(x, y, c='r')
        
        x = [agent.needs[resources[0]] *
             agent.genome[0] for agent in agents if agent.fitness() >= average_fitness]
        y = [agent.needs[resources[1]] *
             agent.genome[0] for agent in agents if agent.fitness() >= average_fitness]
        ax.scatter(x, y, c='lime')
        
        
        ax.set_title("Scatterplot of metabolic needs times genomic vision")
        ax.set_xlabel("metabolic needs (%s)" % (resources[0]))
        ax.set_ylabel("metabolic needs (%s)" % (resources[1]))
        if seperate_above_mean: pl.legend(['Above average fitness'], ['Below average fitness']);
        pl.ion()  # re-enable interactive plotting
        if show:
            self.show(fig);
        if return_fig:
            return fig, ax  # also add ax to modify figure


class Tracker():
    '''
        Tracker class
        Tracks variables in the model
    '''

    def __init__(self, model=None, filename=None):
        if model is not None:
            self.model = model
            model.trackers.append(self)
        if filename is not None:
            import pickle
            with open(filename, 'rb') as f:
                (self.data, self.trackers) = pickle.load(f)
        else:
            self.data = dict()
            self.trackers = dict()

    def add_tracker(self, name, track_fun=lambda model: len(model.agents)):
        '''
            Adds a tracker

            track_fun is a function to be called every iteration of the model.
            It recieves the model as its only argument and its return value is appended to the data
            under the name specified as [name]

            track_fun: The data gatherer function
        '''
        self.trackers[name] = track_fun
        self.data[name] = []

    def track(self):
        assert(self.model is not None), "Can only track when a model is specified"
        for key, fun in self.trackers.items():
            self.data[key].append(fun(self.model))

    def get_data(self, name=None):
        if name is None:
            return data
        else:
            return data[name]

    def save_data(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump((self.data, self.trackers), f)

    def __getitem__(self, key):
        return self.data[key]

#### Regular help functions ####


def normalize_dict(d):
    factor = 1.0 / sum([v for k, v in d.items()])
    for k in d:
        d[k] = d[k] * factor
