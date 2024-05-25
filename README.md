## Requirement
- Python 3.11
- Pytorch
- Streamlit

## Install dependencies
```
pip install -r requirements.txt
```

## How to run
```
streamlit run main.py
     
```

### Demo question list
- I see code app_w304 on my dashboard what to do?
- I see code app_w222 on my dashboard what to do?
- I see code app_w218 on my dashboard what to do?
- I see code bms_a067 on my dashboard what to do?

## Known issues
Streamlit does not [compat with](https://discuss.streamlit.io/t/problem-importing-module-inside-streamlit-script-intermittent-import-failure/64737) transform package, there would be an error `cannot import name 'AutoTokenizer' from 'transformers'` in the page when start the app. Ignore it and refresh the page.