from ABME import *  # Agent is needed (atleast)


'''
    Regular sugerscape agent with the addition of dual resources
    
    Initial model
        Grid map with two resources

        Agent can harvest resources with an efficiency

        Efficiency, should it be harvest reduction (yield) or percentage harvestable from cell.
        Point 1 leaves no resources on cell, point 2 leaves resources on cell.
        Agent has metabolism and requires:

        Metabolism_rate * normalized_vector
        Where vector is of length 2 (number of resources) and denotes the percentage of resource      needed.
        Resources are ‘grown’ to a capacity determined by us

        Harvested resource are (currently) depleted regardless of efficiency;
        Efficiency is a multiplier for the amount of resources gathered
        e.g.: resource available is 4, efficiency is .25. harvested is 1. Left over is 0
'''


class SugerAgent(Agent):
    # Parameters for agent
    wealth_low = 1  # Random low for all resources seperately
    wealth_high = 3  # random high for all resources (inclusive)
    # Metabolism: The total rate of resouce comsumption
    metabolism_low = 1
    metabolism_high = 4
    # max Age in ticks
    age_low = 10
    age_high = 50
    # harvest efficiency
    h_eff = 1  # for all
    # Vision
    vis_low = 1
    vis_high = 6  # Vision in all directions (vonn neumann)
    # Offspring (see def die())
    off_low = 0
    off_high = 5

    '''
        Describe the agent as a string
    '''

    def __str__(self):
        description = []
        description.append("Agent %i" % self.id)
        description.append("\tMetabolism (%f-%f): %f" %
                           (self.metabolism_low, self.metabolism_high, self.metabolism))
        for k, v in self.needs.items():
            description.append("\t\tneed %f%% %s" % (v * 100.0, k))
        description.append("\tAge (%i-%i): %i (max:%i)" %
                           (self.age_low, self.age_high, self.age, self.max_age))
        description.append("\tVision (%i-%i): %i" %
                           (self.vis_low, self.vis_high, self.vision))
        description.append("\twealth (%f-%f):" %
                           (self.wealth_low, self.wealth_high))
        for name, wealth in self.wealth.items():
            description.append("\t\t%s: %f" % (name, wealth))

        fin_str = ""
        for i in description:
            fin_str += i + "\n"
        return fin_str

    '''
        Fuction to compare two agent and sort them according to their fitness value
    '''

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    '''
        Initialize agent specific variables here
        This function is called when agent is added to model (model.add_agent(...,init_agent=True))
    '''

    def init(self):
        assert(self.model is not None), "Model is None, was agent added to model?"
        assert(self.grid is not None), "Grid is None, Agent was not placed in a grid. This agent requires a grid"

        # Initialize wealth
        self.wealth = dict()
        for r_name in self.grid.resources:
            self.wealth[r_name] = np.random.randint(
                self.wealth_low, self.wealth_high)

        # age
        self.age = 0
        self.max_age = np.random.randint(self.age_low, self.age_high)

        # Metabolism
        self.metabolism = np.random.randint(
            self.metabolism_low, self.metabolism_high)
        self.needs = dict()
        for r_name in self.grid.resources:
            self.needs[r_name] = np.random.random()
        normalize_dict(self.needs)  # make vectorlength 1

        # harvest efficiency
        self.harvest_eff = dict()
        for r_name in self.grid.resources:
            self.harvest_eff[r_name] = 1.0

        # Vision
        self.vision = np.random.randint(self.vis_low, self.vis_high)

    '''
        Define how the agent moves
    '''

    def move(self):
        for k, v in self.wealth.items():
            assert(v >= 0), "Wut? should be dead"
        direction = self.choose_direction()
        self.grid.move_agent(self, move_name=direction,
                             move_to=None, move=None, move_idx=None)

    '''
        The direction utility calculator
    '''

    def movUtility(self, highestResource, distance, resource):
        return (max_val) / (0.001 * self.wealth[resource] + 0.1 * distance)

    def choose_direction(self):
        '''
            Returns a neighbourhood name ('left', 'right', etc)
            of the chosen direction.
            Requires: self.look()
        '''
        vision = self.look()  # dict[resource][direction]

        ### Calculate utility for each direction ###

        # Create a empty vector with a NaN for each direction
        utility_per_direction = np.zeros(len(self.grid.neighbourhood))

        # For each resource: Determine the best option
        # and Create a utility based on the best direction and distance per
        # resource
        for resource in vision:  # Iterate first level keys = resource names

            # Direction selection for a single resource
            (dir_name, distance, max_val) = self.choose_best(vision[resource])

            # Convert direction name to a index (e.g. 'left' -> 0)
            dir_idx = np.where(self.grid.neighbourhood_names == dir_name)

            # Determine utility, if something was found
            if not distance == 9999:
                # Note: Dont be negative!
                utility = self.movUtility(max_val, distance, resource)

                # Tetst utility is not negative
                if utility < 0:
                    utility = 0
                # Add utility to best selection accumulator
                utility_per_direction[dir_idx] += utility

        # Determine highest utility direction
        direction_index = np.argmax(utility_per_direction)

        # If no preferences, take random
        if (utility_per_direction == 0).all():
            return np.random.choice(self.grid.neighbourhood_names)

        # Return the name of the direction
        return self.grid.neighbourhood_names[direction_index]

    def choose_best(self, r_vision):
        '''
            Chooses the best direction for a single resource based on distance and value
            r_vision must be a dict containing directions and their visions.
            e.g.:
                vision = self.look(); #dict[resource][direction]
                (dir_name, distance, max_val) = self.choose_best(vision[resource]);
        '''
        order = copy.copy(self.grid.neighbourhood_names)
        np.random.shuffle(order)

        # Initialize best
        #      (dir_name, distance, max_val)
        best = (np.random.choice(order), 9999, 0)
        # Go over all directions by name in order *order* which is random
        # (shuffle)
        for dir_name in order:
            vision = r_vision[dir_name]

            # Get highest value and its distance
            max_val = np.max(vision)  # highest value
            max_dist = np.argmax(vision)  # distance of found maximum

            # Check if better than the previous
            if best[2] < max_val and best[1] >= max_dist:
                best = (dir_name, max_dist, max_val)

        # Return results
        return best

    '''
        Define what the agent does (except for movement)
    '''

    def act(self):
        # harvest
        self.grid.harvest(self, self.harvest_eff)
        # Metabolise
        self.metabolism = (5 * self.vision / 4)
        for n in self.wealth:
            self.wealth[n] -= self.metabolism * self.needs[n]

        for k, r in self.wealth.items():
            if r < 0:
                self.die()
                return  # Cant die twice

        # Age
        self.age += 1

        # reproduce
        '''
        accumulated_wealth = sum(self.wealth.values());
        if accumulated_wealth > 10:
            offspring = SugerAgent();
            canidate_places= self.model.grid.get_move_canidates(self);
            if len(canidate_places) > 0:
                self.model.add_agent(offspring, position=random.choice(canidate_places));
                for key in self.wealth:
                    self.wealth[key] -= 5;
         '''

    def die(self):
        self.model.remove_agent(self)
        self.model.dead.append(self)


class EvolutionAgent(SugerAgent):
    self.genome = None
    self.genome_if = ['vision', ]

    def __init__(self):
    '''
        Define how the agent moves
    '''

    def move(self):
        for k, v in self.wealth.items():
            assert(v >= 0), "Wut? should be dead"
        direction = self.choose_direction()
        self.grid.move_agent(self, move_name=direction,
                             move_to=None, move=None, move_idx=None)

    '''
        Define what the agent does (except for movement)
    '''

    def act(self):
        # harvest
        self.grid.harvest(self, self.harvest_eff)
        # Metabolise
        self.metabolism = (4 * self.vision / )
        for n in self.wealth:
            self.wealth[n] -= self.metabolism * self.needs[n]

        for k, r in self.wealth.items():
            if r < 0:
                self.die()
                return  # Cant die twice

        # Age
        self.age += 1

        # reproduce
        '''
        accumulated_wealth = sum(self.wealth.values());
        if accumulated_wealth > 10:
            offspring = SugerAgent();
            canidate_places= self.model.grid.get_move_canidates(self);
            if len(canidate_places) > 0:
                self.model.add_agent(offspring, position=random.choice(canidate_places));
                for key in self.wealth:
                    self.wealth[key] -= 5;
         '''

    def die(self):
        self.model.remove_agent(self)
        self.model.dead.append(self)

    def fitness(self):
        fitness = 0
        for k, v in self.wealth.items():
            fitness += v

        return fitness

