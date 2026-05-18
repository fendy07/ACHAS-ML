from hybrid_model import (
    AdaptiveHybridModel
)

model = AdaptiveHybridModel()

samples = [

    # calm
    [72, 34.5, 28, 65, 0.5, 0],
    # walking
    [95, 35.2, 30, 70, 1.2, 1],
    # exercise
    [130, 36.1, 32, 75, 2.5, 2],
    # heavy motion
    [145, 36.5, 33, 80, 4.0, 2],
    # recovery
    [90, 35.4, 29, 65, 1.0, 1],
]

for s in samples:

    result = model.predict(*s)

    print(result)