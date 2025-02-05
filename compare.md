----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1               [-1, 64, 64]           9,984
       BatchNorm1d-2               [-1, 64, 64]             128
              ReLU-3               [-1, 64, 64]               0
            Conv1d-4               [-1, 64, 64]          12,288
       BatchNorm1d-5               [-1, 64, 64]             128
              ReLU-6               [-1, 64, 64]               0
            Conv1d-7               [-1, 64, 64]          12,288
       BatchNorm1d-8               [-1, 64, 64]             128
 AdaptiveAvgPool1d-9                [-1, 64, 1]               0
           Linear-10                    [-1, 4]             256
             ReLU-11                    [-1, 4]               0
           Linear-12                   [-1, 64]             256
          Sigmoid-13                   [-1, 64]               0
    LightSEModule-14               [-1, 64, 64]               0
             ReLU-15               [-1, 64, 64]               0
       BasicBlock-16               [-1, 64, 64]               0
           Conv1d-17              [-1, 128, 32]          24,576
      BatchNorm1d-18              [-1, 128, 32]             256
             ReLU-19              [-1, 128, 32]               0
           Conv1d-20              [-1, 128, 32]          49,152
      BatchNorm1d-21              [-1, 128, 32]             256
AdaptiveAvgPool1d-22               [-1, 128, 1]               0
           Linear-23                    [-1, 8]           1,024
             ReLU-24                    [-1, 8]               0
           Linear-25                  [-1, 128]           1,024
          Sigmoid-26                  [-1, 128]               0
    LightSEModule-27              [-1, 128, 32]               0
           Conv1d-28              [-1, 128, 32]           8,192
      BatchNorm1d-29              [-1, 128, 32]             256
             ReLU-30              [-1, 128, 32]               0
       BasicBlock-31              [-1, 128, 32]               0
           Conv1d-32              [-1, 256, 16]          98,304
      BatchNorm1d-33              [-1, 256, 16]             512
             ReLU-34              [-1, 256, 16]               0
           Conv1d-35              [-1, 256, 16]         196,608
      BatchNorm1d-36              [-1, 256, 16]             512
AdaptiveAvgPool1d-37               [-1, 256, 1]               0
           Linear-38                   [-1, 16]           4,096
             ReLU-39                   [-1, 16]               0
           Linear-40                  [-1, 256]           4,096
          Sigmoid-41                  [-1, 256]               0
    LightSEModule-42              [-1, 256, 16]               0
           Conv1d-43              [-1, 256, 16]          32,768
      BatchNorm1d-44              [-1, 256, 16]             512
             ReLU-45              [-1, 256, 16]               0
       BasicBlock-46              [-1, 256, 16]               0
           Conv1d-47               [-1, 512, 8]         393,216
      BatchNorm1d-48               [-1, 512, 8]           1,024
             ReLU-49               [-1, 512, 8]               0
           Conv1d-50               [-1, 512, 8]         786,432
      BatchNorm1d-51               [-1, 512, 8]           1,024
AdaptiveAvgPool1d-52               [-1, 512, 1]               0
           Linear-53                   [-1, 32]          16,384
             ReLU-54                   [-1, 32]               0
           Linear-55                  [-1, 512]          16,384
          Sigmoid-56                  [-1, 512]               0
    LightSEModule-57               [-1, 512, 8]               0
           Conv1d-58               [-1, 512, 8]         131,072
      BatchNorm1d-59               [-1, 512, 8]           1,024
             ReLU-60               [-1, 512, 8]               0
       BasicBlock-61               [-1, 512, 8]               0
AdaptiveAvgPool1d-62               [-1, 512, 1]               0
          Flatten-63                  [-1, 512]               0
           Linear-64                    [-1, 6]           3,078
AdaptiveAvgPool1d-65               [-1, 512, 1]               0
          Flatten-66                  [-1, 512]               0
           Linear-67                   [-1, 16]           8,208
================================================================
Total params: 1,815,446
Trainable params: 1,815,446
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 1.32
Params size (MB): 6.93
Estimated Total Size (MB): 8.27

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 128, 64]          46,592
       BatchNorm1d-2              [-1, 128, 64]             256
              ReLU-3              [-1, 128, 64]               0
         MaxPool1d-4              [-1, 128, 32]               0
            Conv1d-5              [-1, 128, 32]          49,152
       BatchNorm1d-6              [-1, 128, 32]             256
              ReLU-7              [-1, 128, 32]               0
            Conv1d-8              [-1, 128, 32]          49,152
       BatchNorm1d-9              [-1, 128, 32]             256
             ReLU-10              [-1, 128, 32]               0
       BasicBlock-11              [-1, 128, 32]               0
           Conv1d-12              [-1, 128, 16]          49,152
      BatchNorm1d-13              [-1, 128, 16]             256
             ReLU-14              [-1, 128, 16]               0
           Conv1d-15              [-1, 128, 16]          49,152
      BatchNorm1d-16              [-1, 128, 16]             256
           Conv1d-17              [-1, 128, 16]          16,384
      BatchNorm1d-18              [-1, 128, 16]             256
             ReLU-19              [-1, 128, 16]               0
       BasicBlock-20              [-1, 128, 16]               0
           Conv1d-21               [-1, 256, 8]          98,304
      BatchNorm1d-22               [-1, 256, 8]             512
             ReLU-23               [-1, 256, 8]               0
           Conv1d-24               [-1, 256, 8]         196,608
      BatchNorm1d-25               [-1, 256, 8]             512
           Conv1d-26               [-1, 256, 8]          32,768
      BatchNorm1d-27               [-1, 256, 8]             512
             ReLU-28               [-1, 256, 8]               0
       BasicBlock-29               [-1, 256, 8]               0
           Conv1d-30               [-1, 512, 4]         393,216
      BatchNorm1d-31               [-1, 512, 4]           1,024
             ReLU-32               [-1, 512, 4]               0
           Conv1d-33               [-1, 512, 4]         786,432
      BatchNorm1d-34               [-1, 512, 4]           1,024
           Conv1d-35               [-1, 512, 4]         131,072
      BatchNorm1d-36               [-1, 512, 4]           1,024
             ReLU-37               [-1, 512, 4]               0
       BasicBlock-38               [-1, 512, 4]               0
           Conv1d-39               [-1, 512, 2]         786,432
      BatchNorm1d-40               [-1, 512, 2]           1,024
             ReLU-41               [-1, 512, 2]               0
AdaptiveAvgPool1d-42               [-1, 512, 1]               0
           Linear-43                    [-1, 6]           3,078
           Conv1d-44               [-1, 512, 2]         786,432
      BatchNorm1d-45               [-1, 512, 2]           1,024
             ReLU-46               [-1, 512, 2]               0
AdaptiveAvgPool1d-47               [-1, 512, 1]               0
           Linear-48                   [-1, 16]           8,208
================================================================
Total params: 3,490,326
Trainable params: 3,490,326
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 0.91
Params size (MB): 13.31
Estimated Total Size (MB): 14.25
----------------------------------------------------------------