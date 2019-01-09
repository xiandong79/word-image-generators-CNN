# BNP Assessment Task

I am Xiandong QI with email: xqiad@connect.ust.hk. I chose the following task: 

- An English word image generators, then feed it to machine learning model [preferably neural network] to recognize the word from the image

and finished 3 bonus:

 - Deliver an API to call the program
 - Push the program as a Docker Image to Docker Hub
 - Demonstrate the use of Cloud Computing for the solution



## 1. Quick start

### 1.1 Set up locally

1. Go to current folder 
2. Run 

```bash
pythoh bnp_app.py
```

3. Open another terminal and input (use word "bad" as an example)

```bash
curl --data input_word="bad" http://localhost:5000/predict
```
Then, the prediction result is returned.

```bash
{"pred_texts":["bad"],"top_3_paths":["bad","pbad","kbad"]}
```


### 1.2 Using docker image

```bash
docker pull xiandong/bnp-app
sudo docker run -d -p 5000:5000 xiandong/bnp-app
curl  --data input_word="bad" http://localhost:5000/predict
```

### 1.3 Train the model from scratch

```bash
python train_image_ocr.py
```
And a new folder `image_ocr` with the new model weights will be recorded.
Next, you can replace the model path in `Dockerfile` and `bnp_app.py` as you wish.


## 2. Records of Docker image 

### 2.1 Test in local docker env

#### 2.1.1 Build our Docker container 

run:

```bash
sudo docker build -t bnp-app:latest .
```

#### 2.1.2 Run the Docker container

Now let’s run our Docker container to test our app:

```bash
sudo docker run -d -p 5000:5000 bnp-app
# docker stop <CONTAINER ID> i.e., bd7b477188a5
```

#### 2.1.3 Check the status of your container 

by running

```bash
sudo docker ps -a
```

#### 2.1.4 Debug docker image

```bash
docker logs <CONTAINER ID> i.e., f2672b5aff0a
```

#### 2.1.5 Test our model

open another terminal and input (use "small" as an example)

```bash
curl --data input_word="small" http://localhost:5000/predict
curl  --data input_word="good" http://localhost:5000/predict
curl  --data input_word="am" http://localhost:5000/predict
curl  --data input_word="bad" http://localhost:5000/predict
```

Then, the result is returned.

### 2.2 Push to Docker Hub

#### 2.2.1 Create a Docker Hub

```bash
sudo docker login
sudo docker images
sudo docker tag <IMAGE ID, d03fcc88da88> xiandong/bnp-app
sudo docker push xiandong/bnp-app
```

#### 2.2.2 you can pull image and test

```bash
docker pull xiandong/bnp-app
sudo docker run -d -p 5000:5000 xiandong/bnp-app
curl  --data input_word="bad" http://localhost:5000/predict
```

## 3. The use of Cloud Computing

### 3.1 Create a Kubernetes Cluster

Now we run our docker container in Kubernetes. Note that the image tag is just pointing to our hosted docker image on Docker Hub. In addition, we’ll specify with --port that we want to run our app on port 5000.

```bash
kubectl run bnp-app --image=xiandong/bnp-app --port 5000
# We can verify that our pod is running by typing
kubectl get pods
# anyone visiting the IP address of our deployment can access our API.
kubectl expose deployment bnp-app --type=LoadBalancer --port 80 --target-port 5000
# running
kubectl get service
```

### 3.2 Test in Kubernetes Cluster

```bash
# curl --data input_word="bad" http://localhost:5000/predict
curl  --data input_word="bad" http://<EXTERNAL-IP  of k8s>/predict
```

As you can see below, the API correctly returns the label of beagle for the picture.

```bash
{"pred_texts":["bad"],"top_3_paths":["bad","pbad","kbad"]}
```


## 4. Reference

1. https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
2. https://medium.com/analytics-vidhya/deploy-your-first-deep-learning-model-on-kubernetes-with-python-keras-flask-and-docker-575dc07d9e76
3. https://github.com/Tony607/keras-image-ocr
4. https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
5. https://github.com/tensorflow/tensorflow/issues/14356
6. https://stackoverflow.com/questions/24808043/importerror-no-module-named-scipy
6. https://github.com/nicholastoddsmith/pythonml
7. https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
8. https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
9. [Sequence Modeling
With CTC](https://distill.pub/2017/ctc/)
10. https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
11. https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519


