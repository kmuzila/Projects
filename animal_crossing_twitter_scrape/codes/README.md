# ACNH Data Analysis

### Notes:
* *(Karl)* I am working on an data ingest notebook with examples to pull live data but ran into some parsing issues. For now I am uploading CSVs since the source does not update very often.

    * Below is a snippet of how to create a dictionary of dataframes from the CSVs **(Only test on Windows)**

      ```python
      import pandas as pd
      import os

      acnh = {}
      for x in os.listdir('data'):
          acnh[x[:-4]] = pd.read_csv('data\\' + x)
      ```
* *(Karl)* ~~Similar to above the twint library has a bug in it that when run through jupyter notebook due to the way the library installs. I am working on how to fix it but will at least have csv of the data uploaded soon.~~ It is working now!
* Below is instructions to get twitter data using twint
    ```
    pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
    ```
* There is some problems we cannot control with the backend and twitter connection so I would advise exporting the data when you find something interesting.
#### Basic Twint Query
* This is a baseline query that can be used to get started and test that everything works
    ```python
    import twint
    import nest_asyncio # This is important to run the queries and installs with twint.
                        # if it is not installed run: pip install nest_asyncio

    config = twint.Config()
    config.Search = "#ACNH"
    config.Limit = 20 # Limits the number of tweets pulled (increments of 20)
    config.Lang = "en" # Pull only english tweets (not perfect filters a lot)
    nest_asyncio.apply()
    twint.run.Search(config)
    ```
#### Useful Config Options
* The below query searches for the word birthday and either hashtag
    ```python
    config.Search = "birthday AND (#AnimalCrossing OR #ACNH)"
    ```
* Set the minimum interactions to only find popular tweets instead of most recent
    ```python
    c.Popular_tweets = True
    ```
* Manually set the minimum interaction of the tweet
    ```python
    c.Min_likes = 5
    c.Min_retweets = 4
    c.Min_replies = 3
    ```
* Reduce noise by filtering retweets
    ```python
    c.Filter_retweets = True
    ```
* Output to dataframe
    ```python
    config.Pandas = True

    # Twint store the dataframe at: twint.storage.panda.Tweets_df
    Tweets_df = twint.storage.panda.Tweets_df
    ```

* Output to csv (Needs both lines to output the csv)
    ```python
    config.Store_csv = True
    config.Output = "acnh.csv"
    ```

* Twint has a lot of options to configure our query: [config wiki](https://github.com/twintproject/twint/wiki/Configuration)
