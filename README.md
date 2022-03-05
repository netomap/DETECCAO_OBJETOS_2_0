# DETECCAO_OBJETOS_2_0
 Detecção de objetos

### Teste e aprendizado para classificação e detecção de imagens para um dataset fictício, de 15.000 imagens de triângulo, círculos e vazias.

### O aprendizado é bem simples, com a premissa de que só existe um objeto por imagem. Os próximos repositórios serão uma tentatativa de implementar um modelo de detecção utilizando o conceito da rede YOLO.  

- A parte inicial do modelo segue conforme o código abaixo:  

```
self.backbone = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2),

        )
```
- E sua parte final tem duas saídas:

- A que prevê o bbox do objeto:  
```
self.pbox = nn.Sequential(
            self.backbone,
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=4, bias=True),
            nn.ReLU()
        )
```  

- E a que prevê a classe do objeto:  
```
self.pclasse = nn.Sequential(
            self.backbone,
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
            nn.Softmax(dim=-1)
        )
```  

- As funções perda são:  
```
criterion_bbox = nn.MSELoss()
criterion_target = nn.CrossEntropyLoss()
```

- As atualizações dos pesos são feita conforme:  
```
model.zero_grad()

pred_bbox, pred_target = model(batch_tensor)

loss_bbox = criterion_bbox(pred_bbox, batch_bbox)
loss_target = criterion_target(pred_target, batch_target)

total_loss = loss_bbox + loss_target
total_loss.backward()

optimizer.step()
```

- E otimizador:  
```
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)]
```

- Os resultados do treinamento para classificação do objeto são:  
![xxx](./imgs_result/target_train_loss.png)
![xxx](./imgs_result/target_test_loss.png)  

- E para detecção dos objetos:  
![xxx](./imgs_result/bbox_train_loss.png)
![xxx](./imgs_result/bbox_test_loss.png)

- E acurácia do modelo:  
![xxx](./imgs_result/accuracy_test.png)

# E por último uma amostra do resultado final das classificações e detecções:  
- O texto e detecção em verde representam as classificações e bbox **reais** enquanto que o texto e detecção em vermelho representam a **predição** do modelo. **Prob** representa a confiança da classificação, que no caso é 100% para todas essas imagens. (Isso ocorreu devido a simplicidade do dataset).  
![resultado](./imgs_result/output.png)

- Com certeza os resultados podem ser melhores, ainda mais que é um dataset bem simples, porém podemos ver a implementação de uma rede neural capaz de detectar objetos e classificá-los.