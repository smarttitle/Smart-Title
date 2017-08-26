```python
import json
dataset = json.load(open('dataset.json))
```

categories = ['politics', 'business', 'entertainment', 'sport', 'tech']

politics_articles = dataset['politics']

sport_articles = dataset['sport']
...