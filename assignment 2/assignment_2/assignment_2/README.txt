# README for examples shown in course "Reinforcement Learning for Business, Economics, and Social Sciences", Davud Rostam-Afschar (University of Mannheim) based on AI projects developed at UC Berkeley, http://ai.berkeley.edu.

# Make sure you have installed python. Open the folder "assignment_2" in the command line interface.



python gridworld.py -m

# The blue dot is the agent. Note that when you press up, the agent only actually moves north 80% of the time. Such is the life of a Gridworld agent!



python gridworld.py -h

# Options:
#   -h, --help            show this help message and exit
#   -d DISCOUNT, --discount=DISCOUNT
#                         Discount on future (default 0.9)
#   -r R, --livingReward=R
#                         Reward for living for a time step (default 0.0)
#   -n P, --noise=P       How often action results in unintended direction
#                         (default 0.2)
#   -e E, --epsilon=E     Chance of taking a random action in q-learning
#                         (default 0.3)
#   -l P, --learningRate=P
#                         TD learning rate (default 0.5)
#   -i K, --iterations=K  Number of rounds of value iteration (default 10)
#   -k K, --episodes=K    Number of epsiodes of the MDP to run (default 1)
#   -g G, --grid=G        Grid to use (case sensitive; options are BookGrid,
#                         BridgeGrid, CliffGrid, MazeGrid, default BookGrid)
#   -w X, --windowSize=X  Request a window width of X pixels *per grid cell*
#                         (default 150)
#   -a A, --agent=A       Agent type (options are 'random', 'value' and 'q',
#                         default random)
#   -t, --text            Use text-only ASCII display
#   -p, --pause           Pause GUI after each time step when running the MDP
#   -q, --quiet           Skip display of any learning episodes
#   -s S, --speed=S       Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0
#                         is slower (default 1.0)
#   -m, --manual          Manually control agent
#   -v, --valueSteps      Display each step of value iteration




# Example from lecture maze (use arrows to run):

python gridworld.py -a q -k 100 -g BookGrid --noise 0.2 -m -w 250 -r -0.04 -d 1 -l 0.5

# Example from lecture maze with value iteration (no learning):

python gridworld.py -a value -i 100 -g BookGrid --noise 0.2 -w 250 -r -0.04 -d 1


# Example from lecture maze with Epsilon Greedy (epsilon=10%): Your final Q-values should resemble those of your value iteration agent, especially along well-traveled paths. However, your average returns will be lower than the Q-values predicted because of the random actions and the initial learning phase.

python gridworld.py -a q -k 100 -g BookGrid --noise 0.2 -e 0.1 -w 250 -r -0.04 -d 1   -l 0.5

# Example from lecture maze with Epsilon Greedy (epsilon=90%):

python gridworld.py -a q -k 100 -g BookGrid --noise 0.2 -e 0.9 -w 250 -r -0.04 -d 1   -l 0.5





# Run a Q-learning crawler robot and play around with the various learning parameters to see how they affect the agent’s policies and action: step delay is a parameter of the simulation, whereas the learning rate and epsilon are parameters of your learning algorithm, and the discount factor is a property of the environment.

python crawler.py





# Bridge Crossing: BridgeGrid is a grid world map with a low-reward terminal state and a high-reward terminal state separated by a narrow “bridge”, on either side of which is a chasm of high negative reward. The agent starts near the low-reward state. With the default discount of 0.9 and the default noise of 0.2, the optimal policy does not cross the bridge. At what value of discount or noise parameters will the the agent attempt to cross the bridge?

# Value iteration (no learning):

python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

python gridworld.py -a value -i 50 -g BridgeGrid  --discount 0.9 -n 0 


# Train a completely random Q-learner (-e 1) with the default learning rate on the noiseless BridgeGrid for 50 episodes and observe whether it finds the optimal policy.

python gridworld.py -a q -k 50 -g BridgeGrid --discount 0.9 -n 0 -e 1

# Now try the same experiment with an epsilon of 0. Is there an epsilon and a learning rate for which it is highly likely (greater than 99%) that the optimal policy will be learned after 50 iterations?
python gridworld.py -a q -k 50 -g BridgeGrid --discount 0.9 -n 0 -e 0

# This takes long:

python gridworld.py -a q -k 1000 -g BridgeGrid --discount 1 -n 0 -e 1 -l 0.8




# Time to play some Pacman! Pacman will play games in two phases. In the first phase, training, Pacman will begin to learn about the values of positions and actions. Because it takes a very long time to learn accurate Q-values even for tiny grids, Pacman’s training games run in quiet mode by default, with no GUI (or console) display. Once Pacman’s training is complete, he will enter testing mode. When testing, Pacman’s self.epsilon and self.alpha will be set to 0.0, effectively stopping Q-learning and disabling exploration, in order to allow Pacman to exploit his learned policy. Test games are shown in the GUI by default. If you want to experiment with learning parameters, you can use the option -a, for example -a epsilon=0.1,alpha=0.3,gamma=0.7.

python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# If you want to watch 10 training games to see what’s going on, use the command:

python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

# During training, you will see output every 100 games with statistics about how Pacman is faring. Epsilon is positive during training, so Pacman will play poorly even after having learned a good policy: this is because he occasionally makes a random exploratory move into a ghost. As a benchmark, it should take between 1000 and 1400 games before Pacman’s rewards for a 100 episode segment becomes positive, reflecting that he’s started winning more than losing. By the end of training, it should remain positive and be fairly high (between 100 and 350).

# Make sure you understand what is happening here: the MDP state is the exact board configuration facing Pacman, with the now complex transitions describing an entire ply of change to that state. The intermediate game configurations in which Pacman has moved but the ghosts have not replied are not MDP states, but are bundled in to the transitions.

# Once Pacman is done training, he should win very reliably in test games (at least 90% of the time), since now he is exploiting his learned policy.

# However, you will find that training the same agent on the seemingly simple mediumGrid does not work well. In our implementation, Pacman’s average training rewards remain negative throughout training. At test time, he plays badly, probably losing all of his test games. Training will also take a long time, despite its ineffectiveness.

# Pacman fails to win on larger layouts because each board configuration is a separate state with separate Q-values. He has no way to generalize that running into a ghost is bad for all positions. Obviously, this approach will not scale.




# Approximate Q-Learning: Implement an approximate Q-learning agent that learns weights for features of states, where many states might share the same features. ApproximateQAgent uses the IdentityExtractor, which assigns a single feature to every (state,action) pair.

python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid

# Once you’re confident that your approximate learner works correctly with the identity features, run your approximate Q-learning agent with our custom feature extractor, which can learn to win with ease:

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid

# Even much larger layouts should be no problem for your ApproximateQAgent (warning: this may take a few minutes to train). Your approximate Q-learning agent should win almost every time with these simple features, even with only 50 training games:

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic


#####

# Interesting files:
# - valueIterationAgents.py: A value iteration agent for solving known MDPs.
# - qlearningAgents.py: Q-learning agents for Gridworld, Crawler and Pacman.
# - analysis.py: A file to put your answers to questions given in the project.

# Also interesting:
# - mdp.py: Defines methods on general MDPs.
# - learningAgents.py: Defines the base classes ValueEstimationAgent and QLearningAgent, which your agents will extend.
# - util.py: Utilities, including util.Counter, which is particularly useful for Q-learners.
# - gridworld.py: The Gridworld implementation.
# - featureExtractors.py: Classes for extracting features on (state, action) pairs. Used for the approximate Q-learning agent (in qlearningAgents.py).

# Less interesting files:

# - environment.py: Abstract class for general reinforcement learning environments. Used by gridworld.py.
# - graphicsGridworldDisplay.py: Gridworld graphical display.
# - graphicsUtils.py: Graphics utilities.
# - textGridworldDisplay.py: Plug-in for the Gridworld text interface.
# - crawler.py: The crawler code and test harness
# - graphicsCrawlerDisplay.py: GUI for the crawler robot.
# - autograder.py: Project autograder.
# - testParser.py: Parses autograder test and solution files.
# - testClasses.py: General autograding test classes.
# - test_cases/: Directory containing the test cases for each question.
# - reinforcementTestClasses.py: Project 3 specific autograding test classes.

