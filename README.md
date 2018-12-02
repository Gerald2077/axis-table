# PROJECT: Tablesmith 
> Might turn this into SaaS to decode tabular data from any document

### Team: Cyberpunk 

### Theme: Table reading and Understanding in Documents/Images

### Name: Rohit Sharma, sharmarohit2077@gmail.com 

### Youtube: [https://youtu.be/OR0nF4hKILI](https://youtu.be/OR0nF4hKILI)

# Instructions 

* Clone/Download this repository, then cd into it

* Install Required Packages like Python 3.6, then run ``` pip install -r requirements.txt ```

* Download the latest weight file from url in logs/weight-file-download-link.md & place it in the logs folder

* To run the app ``` python app.py  ```

* Now the app is running on http://localhost:8080/ which opens a web page to upload a file and generate a link to download when it is ready.

* You can also make REST api call using curl or postman, 

    ``` curl -F file=@/some/file/on/your/local/disk http://localhost:8080/table -o 1.xls ```
    
    ``` curl -i -X POST -H "Content-Type: multipart/form-data" -F "data=@test.mp3" http://localhost:8080/table -o 1.xls ```

    ``` curl https://jsonplaceholder.typicode.com/todos/1 -o 1.xls ```
