# Back propogation

The objective is train a neural network in excel and show back propogation working on weights

# Proof of work

![alt text](https://github.com/senthilva/EVA6/blob/main/session4/nn%20in%20excel.png "Neural Network")

## Network considered

![alt text](nn.png "Neural Network")


# Calculations

* h1 = w1*i1 + w2*i2	
* h2 = w3*i1 + w4*i2	
* a_h1 = sigmoid(h1) = 1/(1+ exp(-h1))	
* a_h2 = sigmoid(h2) = 1/(1+ exp(-h2))	
* o1 = w5*a_h1 + w6*a_h2	
* o2 = w7*a_h1 + w8*a_h2	
* a_o1 = sigmoid(o1) = 1/(1+ exp(-o1))	
* a_o2 = sigmoid(o2) = 1/(1+ exp(-o2))	
* E_total = E1+ E2	
* E1 = 1/2*(t1-a_o1)^2	
* E1 = 1/2*(t2-a_o2)^2	
*
* ðE1/ða_h1 = a_01-t1*a_o1*(1-ao1)* a_h1*w5					
* ðE2/ða_h1 = a_02-t2*a_o2*(1-ao2)* a_h1*w7					
* ðE_total/ða_h1 = (a_01-t1)*a_o1*(1-ao1)**w5+ (a_02-t2)*a_o2*(1-ao2)* w7					
* ðE_total/ða_h2 = (a_01-t1)*a_o1*(1-ao1)* w6 + (a_02-t2)*a_o2*(1-ao2)* w8		
*
* ðE_total/ðw5 = ð(E1+E2)/ðw5	
* ðE_total/ðw5 = ð(E1)/ðw5	
* ðE_total/ðw5 = ð(E1)/ðw5= ðE1/ða_o1*ða_o1/ðo1*ðo1/ðw5	
* ðE1/ða_o1 = ð(1/2*(t1-a_o1)^2)/ða_o1 = -1(t1-a_o1) = a_01-t1	
* ða_o1/ðo1 = ð(1/(1+ exp(-o1)))/ðo1 = a_o1*(1-ao1)	
* ðo1/ðw5 = a_h1	
*
* ðE_total/ðw5 = (a_01-t1)*a_o1*(1-ao1)* a_h1			
* ðE_total/ðw6 = (a_01-t1)*a_o1*(1-ao1)* a_h2			
* ðE_total/ðw8 = (a_02-t2)*a_o2*(1-ao2)* a_h2			
* ðE_total/ðw7 = (a_02-t2)*a_o2*(1-ao2)*a_h1	
* 		
* ðE_total/ðw1 =a_01-t1*a_o1*(1-ao1)* a_h1*(1-ah1)*i1				
* ðE_total/ðw2 =a_01-t1*a_o1*(1-ao1)* a_h1*(1-ah1)*i2				
* ðE_total/ðw4 =a_02-t2*a_o2*(1-ao2)* a_h2*(1-ah2)*i2				
* ðE_total/ðw3 =a_02-t2*a_o2*(1-ao2)* a_h2*(1-ah2)*i1			
*
* ðE_total/ðw1 =ðE_total/ða_h1*ða_h1/ðh1*ðh1/w1 =  ((a_01-t1)*a_o1*(1-ao1)*w5+ (a_02-t2)*a_o2*(1-ao2)* w7)*a_h1*(1-a_h1)*i1										
* ðE_total/ðw2 =ðE_total/ða_h1*ða_h1/ðh1*ðh1/w2 =  ((a_01-t1)*a_o1*(1-ao1)*w5+ (a_02-t2)*a_o2*(1-ao2)* w7)*a_h1*(1-a_h1)*i2										
* ðE_total/ðw4=ðE_total/ða_h2*ða_h2/ðh2*ðh2/w4 =  ((a_01-t1)*a_o1*(1-ao1)*w6 + (a_02-t2)*a_o2*(1-ao2)* w8)*a_h2*(1-ah2)*i2										
* ðE_total/ðw3=ðE_total/ða_h2*ða_h2/ðh2*ðh2/w3 =  ((a_01-t1)*a_o1*(1-ao1)*w6 + (a_02-t2)*a_o2*(1-ao2)* w8)*a_h2*(1-ah2)*i1										

# Excel Calculations

![image](https://user-images.githubusercontent.com/8141261/120002314-5260d280-bff2-11eb-9f4d-d541ed350c14.png)

# Error vs LR

![image](https://user-images.githubusercontent.com/8141261/120002506-8936e880-bff2-11eb-9f95-c1cb56b3d21e.png)


# Error vs LR graph

![image](https://user-images.githubusercontent.com/8141261/120002666-b1264c00-bff2-11eb-8283-8146576125a7.png)

## Observations
 
* As LR increases it converges faster - as it takes larger steps 
