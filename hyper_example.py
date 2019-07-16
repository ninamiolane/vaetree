import ray
from ray import tune

ray.init()

def train_func(config, reporter):  # add a reporter arg
     model = ( ... )
     optimizer = SGD(model.parameters(),
                     momentum=config["momentum"])
     dataset = ( ... )

     for idx, (data, target) in enumerate(dataset):
         accuracy = model.fit(data, target)
         reporter(mean_accuracy=accuracy) # report metrics


all_trials = tune.run(
    train_func,
    name="quick-start",
    stop={"mean_accuracy": 99},
    config={"momentum": tune.grid_search([0.1, 0.2])}
)
