# Recommender-Systems HW: 2

ID: 2020223040013 <br/>
Name: 邓梓烽

## 1. Abstract

&#8195;This is my second article-reproducing homework. However, I must confess that I have not fully figured out the mechanism of this whole program (or, a *clear* and *thorough* understanding of the *heart* of this article (eg. GAN)), and thus, all I can do is fetching the [source program](https://github.com/changfengsun/LARA) provided by the authors of this article and try it out. <br/>
&#8195;However, the source code provided by the authors is not as readable as I thought it would be (the part of visualizing the metrices mentioned in the article is even wiped out in this public version!), and the number of training round is almost twice as many as the article shown. <br/>
&#8195;After adding some missing visualization codes, the results given in the graphs are sightly different from the original results given in the article, and I have not yet figure out what the true reason is. <br/>
+ Could it be blamed to the randomized ininitial setting?
+ Could it be possible that the settings mentioned in the article is slightly different from the ones given in this source code?

> By the way, to speed up the training period, I have already reduce the number of training rounds in the source code. I don't know why the number of training times differ in the source code mentioned above and the one shown in the graphs of the article.

<br/>
<br/>

## 2. Run-time snip and visualization of the metrices

&#8195;The learning curves will be given in the following, and the scales of both axes have been set to be almost the same as the original graphs shown in the article.<br/>

![movielens_runtime_1](https://user-images.githubusercontent.com/82326445/116053946-c6375480-a6ad-11eb-8c6b-382fb6048ea4.png)

> Run-time snip on movielens.

![movielens_results_1](https://user-images.githubusercontent.com/82326445/116054027-da7b5180-a6ad-11eb-9d28-2fcf43f510b7.png)

![movielens_results_2](https://user-images.githubusercontent.com/82326445/116054044-df400580-a6ad-11eb-8699-4b8d7df3d10c.png)

> Metrices on movielens. As we can see, they are not that smooth as the curve given in the article, and surge at the end of our training round.

![izone_runtime_1](https://user-images.githubusercontent.com/82326445/116054106-ee26b800-a6ad-11eb-9926-41a46ef27151.png)

> Run-time snip on inzone.

![izone_result_1](https://user-images.githubusercontent.com/82326445/116054227-0c8cb380-a6ae-11eb-98f5-f8e9b89b32e0.png)

![izone_result_2](https://user-images.githubusercontent.com/82326445/116054249-0eef0d80-a6ae-11eb-8bc5-d38a439e16ac.png)

> Metrices on inzone. They looks almost the same as the ones given in the article.

<br/>
<br/>

## 3. How to reproduce the results

> &#8195;This is the most important part while the authors chose not to explain in details... : (

1. Install Anaconda (lastest version can work).
2. Create a new env branch, using python 3.6.
3. Open Anaconda's prompt window.
4. Switch to your active env branch using `activate 'branch name'`
5. Install tensorflow-gpu=1.12 using
  + `conda search tensorflow`
  + `conda install tensorflow-gpu=1.12.0`
6. Install pandas, matplotlib on Anaconda's UI.
7. Run "model.py" accordingly.
8. **Wait for at least 45 mins and find something else to do, try not to be crazy**.

