# Project 3: Understanding User Behavior

#### **Joe Damisch, Karl Eirich, Kasha Muzila**

<br>

The projects intention is to understand why Zillow purchases specific properties. We are combining Zillow property data with Yelp data to give insight into the neighborhoods. The goal is to identify trends in the properties features and comercial products or services which we can provide to interested parties.

<br>

## **Pipeline Components**

### **Apache2 Utils**  

Apache2 Utils is used to test HTTP servers using the Apache Benchmark Tool.

### **APT**  

APT is short for Advanced Package Tool and is used for managing software installations on Linux.

### **Beautiful Soup**  

Beautiful Soup is a python library used to parse HTML files.

### **cp**  

cp is a command line tool used for copying files on Unix systems.

### **cURL**  

cURL is a Command line tool used to transfer data using networks and URLs.

### **Datetime**  

Datetime is a python library used to manage date and time data.

### **Docker**  

Docker is a platform allowing the user application in seperate loosely isolated containers. The containers are contain everything to run, test, and deploy the pipeline without depending on external infrastructure.

### **Docker Compose**  

Docker Compose is a tool used to define and run docker containers using YAML files to configure, create, and start services to create the pipeline.

### **Flask**  

Flask is a web framework written in Python.

### **Google Cloud SDK**  

The gcloud tool is used for local tool for managing remote cloud instances and creating remote connections.

### **Git**  

Git is a version control tool for managing local and remote code repositories.

### **Json**  

Json python library is used to parse json files.

### **Kafka**  

Kafka is a distributed event streaming platform used to capture and route data from events to specified destinations.

### **Numpy**  

Numpy is a python library used for scientific computing.

### **Pandas**  

Pandas is a python library built for data analysis, manipulation, and indexing.

### **Pip**  

Pip is a package installation tool for Python.

### **Python**  

Python is a high level programming language.

### **Requests**  

Requests is a python library used to make HTTP requests.

### **Software Properties Common**  

Software properties Common is a linux package used to software distribution repositories

### **Spark**  

Spark is an analytics engine that supports multiple languages including python and SQL used for structured data processing.

### **PySpark**  

PySpark is an interface for Spark in Python allowing for accessing Spark application using Python APIs.

### **Spark SQL**  

Spark SQL is a Spark module providing more information about the structure of the data and computations when performing structured data processing.

### **Time**  

A python library used to manage time related functions.

### **Update Alternatives**  

Update Alternatives manages symbolic links on Linux systems.

### **YAML**  

YAML is short for YAML Ain't Markup Language and is used for creating configuration files.

### **Zookeeper**  

Zookeeper is a service for maintaing distributed services and corresponding information.

<br>

## **Pipeline Setup**

1. Clone the repository and create a branch

> `git clone https://github.com/mids-w205-de-sola/w205_project_3_karl_joe_kasha.git`  
> `git branch assignment`  
> `git checkout assignment`

2. Copy the docker compose YAML file 

> `cp ~/w205/course-content-fall2021/14-Patterns-for-Data-Pipelines/docker-compose.yml .`

3. Run the following in the terminal to ensure that the Cloudera image can be run correctly

```sh
sudo -s
echo 'GRUB_CMDLINE_LINUX_DEFAULT="vsyscall=emulate"' >> /etc/default/grub
update-grub
reboot
```

4. Start Docker and check if containers are running

> `docker-compose up -d`  
> `docker-compose ps`

5. Run shell script inside docker container to update python version

> `docker-compose exec mids sh /w205/w205_project_3_karl_joe_kasha/setup.sh`

<br>

---

### **Script Content**

1. Download the gpg key to enable connection linux distribution servers

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```

2. Get list of updates available for linux instance and install Apache Bench for testing Flask app

```bash
sudo apt-get update
sudo apt-get install --yes apache2-utils
```

3. Install software properties commmon to enable the use of the deadsnakes distribution server to get python version compatible with our scripts

```bash
sudo apt-get install --yes software-properties-common
sudo add-apt-repository --yes ppa:deadsnakes/ppa 
sudo apt update
```

4. Install Python 3.9 from the deadsnakes server and virtual environment python tools to aid setup

```bash
sudo apt install --yes python3.9
sudo apt-get install --yes python3.9-venv
```

5. Update the python symbolic link to the newly installed python version

```bash
update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2
```

6. Install pip and all the python libraries used in our scripts

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install datetime
python -m pip install requests
python -m pip install pandas
python -m pip install bs4
python -m pip install flask
python -m pip install pyspark
python -m pip install kafka-python
```

---

### **Docker, Kafka, and Flask Commands**

1. Check the docker instance python version

> `docker-compose exec mids python --version`

2. Create Kafka topic called events to log user events

> `docker-compose exec kafka kafka-topics --create --topic events --partitions 1 --replication-factor 1 --if-not-exists --zookeeper zookeeper:32181`

3. Run Flask app

> `docker-compose exec mids env FLASK_APP=/w205/w205_project_3_karl_joe_kasha/app.py flask run --host 0.0.0.0`

4. Continuously consume Kafka events (run in a seperate terminal window)

> `docker-compose exec mids kafkacat -C -b kafka:29092 -t events -o beginning`

5. Run the Spark streaming job

> `docker-compose exec spark spark-submit /w205/w205_project_3_karl_joe_kasha/aggregate_spark_job.py`

6. Check to ensure that a zipcode_data directory exists in the Hadoop tmp folder

> `docker-compose exec cloudera hadoop fs -ls /tmp`

7. Navigate to Hive instance to build external parquet tables

> `docker-compose exec cloudera hive`

8. Create external parquet tables

```sh
create external table if not exists zipcode_data (
    raw_event string,
    timestamp string,
    Accept string,
    `Content-Length` string,
    `Content-Type` string,
    Host string,
    `User-Agent` string,
    zipcodes string,
    event_data string,
    `event_type` string,
    query_timestamp string
)
stored as parquet
location '/tmp/zipcode_data'
tblproperties ("parquet.compress"="SNAPPY");
```

9. Open the Presto prompt

> `docker-compose exec presto presto --server presto:8080 --catalog hive --schema default`

10. Check to see that your table(s) were created

> `show tables;`
> `describe table <table>;`

<br>

---

### **Testing**  

Tests the default Flask app response

```sh
docker-compose exec mids ab -n 10 -H "Host: user1.comcast.com" http://localhost:5000/
```

Tests the zipcode index response (returns list of available zipcodes)

```sh
docker-compose exec mids ab -n 10 -H "Host: user1.comcast.com" http://localhost:5000/zipcode
```

Tests the summary response (returns aggregate statistics of the entire dataset)

```sh
docker-compose exec mids ab -n 10 -H "Host: user1.comcast.com" http://localhost:5000/summary
```

Tests a specific zipcode response (returns aggregate statistics for a specific zipcode)

```sh
docker-compose exec mids ab -n 10 -H "Host: user1.comcast.com" http://localhost:5000/<zipcode>
```

Tests refresh response (returns new data file name)

```sh
docker-compose exec mids ab -n 10 -H "Host: user1.comcast.com" http://localhost:5000/<zipcode>/refresh/<api_key>
```

---

### **Finishing Up**  

Turn down the docker instances

> `docker-compose down`