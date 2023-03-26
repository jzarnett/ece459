#!/usr/bin/env python
# coding: utf-8

# # Notebook Intentions
# 
# The purpose of this notebook is to simulate an amusement park. 
# 

# In[1]:


from park import Park
from behavior_reference import BEHAVIOR_ARCHETYPE_PARAMETERS


# ### Book Keeping
# 
#     - Version: dictates the location performance graphs are stored
#     - Verbosity: controls much information is displayed during a simulation, increase to show more info
#     - Show Plots: controls whether park plots are displayed in this notebook
#     - RNG Seed: seeds random number generators to enforce identical results across runs/machines
#     

# In[2]:


VERSION = "20K All FP - MIKE"
VERBOSITY = 0
SHOW_PLOTS = True
RNG_SEED = 5


# ## Park
#  
# The park contains Agents, Attractions and Activities.
# 
# - Total Daily Agents: dictates how many agents visit the park within a day
# - Hourly Percent: dictates what percentage of Total Daily Agents visits the park at each hour
# - Perfect Arrivals: enforces that the exact amount of Total Daily Agents arrives during the day
# - Expedited Pass Ability Percent: percent of agents aware of expeditied passes
# - Expedited Threshold: acceptable queue wait time length before searching for an expedited pass
# - Expedited Limit: total number of expedited pass an agent can hold at any given time
# 

# In[3]:


TOTAL_DAILY_AGENTS = 2000
PERFECT_ARRIVALS = True
HOURLY_PERCENT = {
    "10:00 AM": 10,
    "11:00 AM": 20,
    "12:00 AM": 17,
    "3:00 PM": 20,
    "4:00 PM": 15,
    "5:00 PM": 10,
    "6:00 PM": 1,
    "7:00 PM": 5,
    "8:00 PM": 1,
    "9:00 PM": 1,
    "10:00 PM": 0,
    "11:00 PM": 0,
    "12:00 PM": 0
}
EXP_ABILITY_PCT = 1.0
EXP_THRESHOLD = 1
EXP_LIMIT = 3


# ## Agents
# 
# The park is populated by agents. Agents visit attractions and activities. They make decisions about where to go based on their preferences, the popularity of attractions and activities and waiting times. Each Agent has a behavioral archetype.
# 
# - Ride Enthusiast: wants to stay for a long time, go on as many attractions as possible, doesn't want to visit activites, doesn't mind waiting
# - Ride Favorer: wants to go on a lot of attractions, but will vists activites occasionally, will wait for a while in a queue
# - Park Tourer: wants to stay for a long time and wants to see attractions and activities equally, reasonable about wait times
# - Park Visitor: doesn't want to stay long and wants to see attractions and activities equally, inpatient about wait times
# - Activity Favorer: doesn't want to stay long and prefers activities, reasonable about wait times
# - Activity Enthusiast: wants to visit a lot of activities, reasonable about wait times
# 
# Archetypes can be tweaked and new archetypes can be added in `behavior_reference.py`. 
# 
# - Agent Archetype Distribution: dictates the probability an agent will have a particular behavioral archetype
# 

# In[4]:


AGENT_ARCHETYPE_DISTRIBUTION = {
    "ride_enthusiast": 10,
    "ride_favorer": 15,
    "park_tourer": 25,
    "park_visitor": 30,
    "activity_favorer": 15,
    "activity_enthusiast": 5,
}


# ## Attractions
# 
# Attractions are essentially rides within a park. 
# 
# - Attractions: list of dictionaries that describe the attractions within the park
#     - Name: name of the attraction
#     - Run Time: How long the attraction take to run, in minutes
#     - Capacity: Maximium number of agents that can be on the attraction during a run
#     - Popularity: Value from 1-10 that describes how popular an attraction is with respect to other attractions
#     - Expedited Queue: Dictates whether the attraction has an expedited queue or not
#     - Expedited Queue Ratio: Dictates what percentage of attraction capacity is devoted to members of the expedited queue
#     - Child Eligible: Dictates whether children can ride the attraction
#     - Adult Eligible: Dictates whether adults can ride the attraction

# In[5]:


ATTRACTIONS = [
    {
        "name": "Alpha",
        "run_time": 10,
        "hourly_throughput": 3000,
        "popularity": 10,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": True,
        "adult_eligible": True,
    },
    {
        "name": "Beta",
        "run_time": 5,
        "hourly_throughput": 2400,
        "popularity": 9,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": True,
        "adult_eligible": True,
    },
    {
        "name": "Gamma",
        "run_time": 15,
        "hourly_throughput": 2000,
        "popularity": 8,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": True,
        "adult_eligible": True,
    },
    {
        "name": "Delta",
        "run_time": 5,
        "hourly_throughput": 1200,
        "popularity": 7,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": True,
        "adult_eligible": False,
    },
    {
        "name": "Epsilon",
        "run_time": 10,
        "hourly_throughput": 2000,
        "popularity": 6,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": False,
        "adult_eligible": True,
    },
    {
        "name": "Zeta",
        "run_time": 6,
        "hourly_throughput": 2000,
        "popularity": 5,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": True,
        "adult_eligible": False,
    },
    {
        "name": "Eta",
        "run_time": 12,
        "hourly_throughput": 2400,
        "popularity": 4,
        "expedited_queue": True,
        "expedited_queue_ratio": 0.8,
        "child_eligible": False,
        "adult_eligible": True,
    }
]


# ## Activities
# Activities are everything to do within the park that isn't an attraction. 
# 
# - Activities: list of dictionaries that describe activities within the park
#     - Name: name of the activity
#     - Popularity: Value from 1-10 that describes how popular an activity is with respect to other activities
#     - Mean Time: The expected time agents will spend at an activity

# In[6]:


ACTIVITIES = [
    {
      "name": "sightseeing",
      "popularity": 5,
      "mean_time": 5
    },
    {
      "name": "show",
      "popularity": 5,
      "mean_time": 30
    },
    {
      "name": "merchandise",
      "popularity": 5,
      "mean_time": 30
    },
    {
      "name": "food",
      "popularity": 5,
      "mean_time": 45
    }
  ]


# ## Plots
# 
# Set the Y limit of plots

# In[7]:


PLOT_RANGE = {
    "Attraction Queue Length": 1800,
    "Attraction Wait Time": 200,
    "Attraction Expedited Queue Length": 6000,
    "Attraction Expedited Wait Time": 500,
    "Activity Vistors": 20000,
    "Approximate Agent Distribution (General)": 1.0,
    "Approximate Agent Distribution (Specific)": 1.0,
    "Attraction Average Wait Times": 120,
    "Agent Attractions Histogram": 1.0,
    "Attraction Total Visits": 46000,
    "Expedited Pass Distribution": 150000,
    "Age Class Distribution": 20000,
}


# ## Simulation
# 
# Run the simulation here.

# In[ ]:


# Initialize Park
RNG_SEED = 5

park = Park(
    attraction_list=ATTRACTIONS,
    activity_list=ACTIVITIES,
    plot_range=PLOT_RANGE,
    random_seed=RNG_SEED,
    version=VERSION,
    verbosity=VERBOSITY
)

# Build Arrivals

park.generate_arrival_schedule(
    arrival_seed=HOURLY_PERCENT, 
    total_daily_agents=TOTAL_DAILY_AGENTS, 
    perfect_arrivals=PERFECT_ARRIVALS,
)

# Build Agents
park.generate_agents(
    behavior_archetype_distribution=AGENT_ARCHETYPE_DISTRIBUTION,
    exp_ability_pct=EXP_ABILITY_PCT,
    exp_wait_threshold=EXP_THRESHOLD,
    exp_limit=EXP_LIMIT
)

# Build Attractions + Activities
park.generate_attractions()
park.generate_activities()

# Pass Time
for _ in range(len(HOURLY_PERCENT.keys()) * 60):
    park.step()

# Save Parameters of Current Run
sim_parameters = {
    "VERSION": VERSION,
    "VERBOSITY": VERBOSITY,
    "SHOW_PLOTS": SHOW_PLOTS,
    "RNG_SEED": RNG_SEED,
    "TOTAL_DAILY_AGENTS": TOTAL_DAILY_AGENTS,
    "PERFECT_ARRIVALS": PERFECT_ARRIVALS,
    "HOURLY_PERCENT": HOURLY_PERCENT,
    "EXP_ABILITY_PCT": EXP_ABILITY_PCT,
    "EXP_THRESHOLD": EXP_THRESHOLD,
    "EXP_LIMIT": EXP_LIMIT,
    "AGENT_ARCHETYPE_DISTRIBUTION": AGENT_ARCHETYPE_DISTRIBUTION,
    "ATTRACTIONS": ATTRACTIONS,
    "ACTIVITIES": ACTIVITIES,
    "BEHAVIOR_ARCHETYPE_PARAMETERS": BEHAVIOR_ARCHETYPE_PARAMETERS,
}
park.write_data_to_file(
    data=sim_parameters, 
    output_file_path=f"{VERSION}/parameters", 
    output_file_format="json"
)

# Store + Print Data
park.make_plots(show=SHOW_PLOTS)
park.print_logs(N = 5)
#park.print_logs(selected_agent_ids = [778])

