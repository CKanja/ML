# Forest Fire Detection Application

[LINK TO DEPLOYED PROJECT](https://fire-detection-app.streamlit.app/)

[LINK TO NOTEBOOK](https://github.com/CKanja/ML/blob/main/Notebook/Machine_Learning_Summative.ipynb)

## Project Overview

### Context
According to Global Forest Watch, a forest monitor that offers the latest data on global forests, between 2001 and 2021, Kenya lost 1.98 kilo hectares (kha) of tree cover from wildfires. Forest fires have severe ripple effects, such as increased carbon emissions; they release substantial amounts of carbon dioxide (CO2) into the atmosphere. This contributes to climate change by adding greenhouse gases to the air, resulting in higher global temperatures and an increased frequency of severe wildfires. Additionally, they affect ecosystems by destroying habitats, causing biodiversity loss, and harming human health, as the smoke poses health risks to nearby communities. 

Forests are often spread nationwide, within cities and in remote areas such as uninhabitable mountainous regions. This geographical dispersion presents significant challenges in detecting and monitoring events like wildfires in real-time. This results in delays in deploying teams to combat the fires. There is a need for remote and real-time monitoring systems to detect signs of wildfire and alert relevant authorities and stakeholders in a timely manner.


### Project Objective
The main objective of this research is to develop and deploy an effective remote sensing and machine learning-powered wildfire detection and alert system in Kenya to enhance early detection and response, ultimately mitigating the impact of wildfires on ecosystems, communities, and wildlife.

### Dataset
The dataset was obtained from [kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset/data) 
The data was divided into 2 folders, fire_images folder contains 755 outdoor-fire images some of them contains heavy smoke, the other one is non-fire_images which contain 244 nature images (eg: forest, tree, grass, river, people, foggy forest, lake, animal, road, and waterfall).

#### Categories:
1. **Forest Fire Images**
2. **Non-Forest Fire Images**


### How to Test
Download the `forest.jpg` or `forest-fire.jpg` images from this repository and uplaod them to the Streamlit application to test. Feel free to also test the application with an image of a forest, to test the model.