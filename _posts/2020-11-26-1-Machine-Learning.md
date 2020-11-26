---
title: 1. 머신 러닝 - Machine Learning
layout: post
categories: ML

use_math: true
---

# 머신 러닝의 정의

Carnegie Mellon Univ. 의 Tom M. Mitchell 교수는 저서 Machine Learning 에서 다음과 같이 정의 했다.

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at task in T, as measured by P, improves with experience E.

>만약 P로 측정된 컴퓨터 프로그램의 작업 T의 성능이 경험 E로 향상되는 경우, 
그 컴퓨터 프로그램은 작업 T와 성능 측정 P에 대해 경험 E를 학습했다라고 할 수 있다.

>- Tom M. Mitchell (1998)

간단하게 어떠한 작업 T에 대해 경험 E를 통하여 그 T에 대한 성능 P를 높이는 것을 **머신 러닝**이라고 할 수 있다.

# 머신 러닝의 학습 방법

머신 러닝의 학습 방법은 크게 3가지로 분류된다.

- **지도 학습 - Supervised Learning**
- **비지도 학습 - Unsupervised Learning**
- **강화 학습 - Reinforcement Learning**

## 지도 학습 - Supervised Learning

![1%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%20%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20-%20Machine%20Learning%20ad2aef2dc74d43148a00cabab8388461/_2020-11-26__9.38.13.png](1%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%20%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20-%20Machine%20Learning%20ad2aef2dc74d43148a00cabab8388461/_2020-11-26__9.38.13.png)

지도 학습은 특정 입력(Input)에 대하여 레이블(label)이 있는 데이터 집합이 주어지는 경우의 학습 방법으로

바꿔 말하면 입력과 레이블(정답)을 함께 주고 학습시키는 방법이다.

예를 들어 위 사진과 같이 사진과 해당 사진의 레이블(cat, dog, ...)을 함께 주고 학습을 시킬 경우 지도 학습에 해당한다.

### 지도 학습의 유형

다음은 지도 학습의 유형과 사용 예시이다.

- **회귀 - Regression**

    ex) 학생의 학습 시간에 따른 시험 점수를 예측하는 모델

- **이진 분류 - Binary Classification**

    ex) 학생의 학습 시간에 따른 시험 합격/불합격 여부를 예측하는 모델

    ex) 사진을 입력 받아 해당 사진이 고양이 사진 인지 판단하는 모델

- **다중 분류 - Multi-Label Classification**

    ex) 학생의 학습 시간에 따른 시험 성적 등급(A, B, C, ...)를 예측하는 모델

    ex) 동물 사진을 입력 받아 해당 사진의 동물 이름(강아지, 고양이, 토끼, ...)을 출력하는 모델

## 비지도 학습 - Unsupervised Learning

![1%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%20%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20-%20Machine%20Learning%20ad2aef2dc74d43148a00cabab8388461/1VACikYaZIHb2OctTRohn8A.jpeg](1%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%20%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20-%20Machine%20Learning%20ad2aef2dc74d43148a00cabab8388461/1VACikYaZIHb2OctTRohn8A.jpeg)

비지도 학습은 특정 입력(Input)에 대하여 레이블(label)이 없는 데이터 집합이 주어지는 경우의 학습 방법으로

입력된 데이터들을 비슷한 특성을 가진 것 끼리 그룹화 할 때 쓰인다.

데이터의 레이블이 주어지지 않았기 때문에 데이터로 부터 패턴이나 특정 형태를 찾아내기가 어려운 방법이지만,

데이터의 숨겨진 특징이나 구조를 결과물로 얻을 수 있다는 점에서 데이터 전처리 방법으로 사용하기도 한다.

예시로 구글 뉴스에서 비슷한 주제의 기사끼리 그룹화 할때 비지도 학습이 사용된다.

## 강화 학습 - Reinforcement Learning

![1%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%20%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20-%20Machine%20Learning%20ad2aef2dc74d43148a00cabab8388461/reinforcement-learning-fig1-700.jpg](1%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%20%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20-%20Machine%20Learning%20ad2aef2dc74d43148a00cabab8388461/reinforcement-learning-fig1-700.jpg)

강화 학습은 지도 학습, 비지도 학습과는 조금 다른 종류의 학습 방법이다.

Agent가 주어진 환경(State)에 대해 어떤 행동(Action)을 취하고, 이로부터 어떤 보상(Reward)을 얻으며 학습을 진행하는데, 이 Agent가 보상을 최대화 하도록 학습을 진행하는 것이다.

쉽게 설명하자면 Agent가 시행착오를 겪으며 보상이 최대화 되도록 학습을 진행한다는 것이다.

예시로 자율 주행 차량에서 강화 학습이 사용된다.