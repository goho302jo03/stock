featurey.py
model.py
  get_product()
  tr = build_model()
    build_data()
      build_x()
      build_y()
  predict(tr, te)
  score()


output format (the same day)
{
    "stock id 1": {
        "feature1": 0,
        "feature2": 0,
        "feature3": 0,
        "feature4": 0,
        "feature5": 0,
    },
    "stock id 2": {
        "feature1": 0,
        "feature2": 0,
        "feature3": 0,
        "feature4": 0,
        "feature5": 0,
    },
}

build_x
input: 
- start day: range of sampling
- end day: range of sampling
- seleced features: type = list, element = 'feature1', 'feature3'
- features days: days per sample

output:
- X: list, [2, 30, 3, ... ], features per day * features day 
