# WhisperLlama

### Whisper Encoder + Llama 4 layers (가장 위쪽 & 가장 아래쪽)

- train_data: trained with librispeech train.100
- epoch: 1 
- lr: 2e-5
- trained model: https://huggingface.co/Kyudan/whisperllama
- training: elice.io 클라우드 A100 1개개 사용 (약 5천원)
- result: 학습이 완전하지 않음 (같은 말 반복)

### Loss
<img src="W&B Chart 2025. 1. 29. 오전 12_30_50.png"  title="loss"/>

### inference test
```
Processing sample 10/10...
Speaker ID: 6930
Chapter ID: 75918
Reference text: IT IS YOU WHO ARE MISTAKEN RAOUL I HAVE READ HIS DISTRESS IN HIS EYES IN HIS EVERY GESTURE AND ACTION THE WHOLE DAY
Running inference...

Using device: cuda

Input features shape: torch.Size([1, 80, 3000])
Using start token: 128000 (<|begin_of_text|>)

Top 5 tokens for position 0
Token: Question, Probability: 0.5033
Token: The, Probability: 0.5033
Token: import, Probability: 0.5033
Token: from, Probability: 0.5033
Token: def, Probability: 0.5033

Top 5 tokens for position 1
Token: Question, Probability: 0.9522
Token:  question, Probability: 0.9522
Token:  Questions, Probability: 0.9522
Token: (Q, Probability: 0.9522
Token:  QUESTION, Probability: 0.9522

Top 5 tokens for position 2
Token: Question, Probability: 0.9754
Token: (Q, Probability: 0.9754
Token: (", Probability: 0.9754
Token: (question, Probability: 0.9754
Token: ([-, Probability: 0.9754

Top 5 tokens for position 3
Token: Question, Probability: 0.9665
Token: (Q, Probability: 0.9665
Token: (', Probability: 0.9665
Token: (question, Probability: 0.9665
Token: ([-, Probability: 0.9665

Top 5 tokens for position 4
Token: Question, Probability: 0.9854
Token: (Q, Probability: 0.9854
Token: (', Probability: 0.9854
Token: (question, Probability: 0.9854
Token: (-, Probability: 0.9854

Top 5 tokens for position 5
Token: Question, Probability: 0.9940
Token: (Q, Probability: 0.9940
Token: .Question, Probability: 0.9940
Token: Indicator, Probability: 0.9940
Token: /question, Probability: 0.9940

Top 5 tokens for position 6
Token: Question, Probability: 0.9954
Token: .Question, Probability: 0.9954
Token: Indicator, Probability: 0.9954
Token: /question, Probability: 0.9954
Token:  QUESTION, Probability: 0.9954

Top 5 tokens for position 7
Token: Question, Probability: 0.9961
Token: .Question, Probability: 0.9961
Token: Indicator, Probability: 0.9961
Token: /question, Probability: 0.9961
Token:  QUESTION, Probability: 0.9961

Top 5 tokens for position 8
Token: Question, Probability: 0.9968
Token: Indicator, Probability: 0.9968
Token: .Question, Probability: 0.9968
Token:  QUESTION, Probability: 0.9968
Token: /question, Probability: 0.9968

Top 5 tokens for position 9
Token: Question, Probability: 0.9971
Token: Indicator, Probability: 0.9971
Token:  QUESTION, Probability: 0.9971
Token: .Question, Probability: 0.9971
Token: /question, Probability: 0.9971

Top 5 tokens for position 10
Token: Question, Probability: 0.9971
Token: Indicator, Probability: 0.9971
Token:  QUESTION, Probability: 0.9971
Token: .Question, Probability: 0.9971
Token: /question, Probability: 0.9971

Top 5 tokens for position 11
Token: Question, Probability: 0.9968
Token: Indicator, Probability: 0.9968
Token:  QUESTION, Probability: 0.9968
Token: .Question, Probability: 0.9968
Token: Questions, Probability: 0.9968

Top 5 tokens for position 12
Token: Question, Probability: 0.9966
Token:  QUESTION, Probability: 0.9966
Token: Indicator, Probability: 0.9966
Token: .Question, Probability: 0.9966
Token: Questions, Probability: 0.9966

Top 5 tokens for position 13
Token: Question, Probability: 0.9972
Token:  QUESTION, Probability: 0.9972
Token: .Question, Probability: 0.9972
Token: Indicator, Probability: 0.9972
Token: Questions, Probability: 0.9972

Top 5 tokens for position 14
Token: Question, Probability: 0.9979
Token:  QUESTION, Probability: 0.9979
Token: .Question, Probability: 0.9979
Token: Indicator, Probability: 0.9979
Token: Questions, Probability: 0.9979

Top 5 tokens for position 15
Token: Question, Probability: 0.9981
Token:  QUESTION, Probability: 0.9981
Token: .Question, Probability: 0.9981
Token: Indicator, Probability: 0.9981
Token: Questions, Probability: 0.9981

Top 5 tokens for position 16
Token: Question, Probability: 0.9980
Token:  QUESTION, Probability: 0.9980
Token: .Question, Probability: 0.9980
Token: Questions, Probability: 0.9980
Token: Indicator, Probability: 0.9980

Top 5 tokens for position 17
Token: Question, Probability: 0.9974
Token:  QUESTION, Probability: 0.9974
Token: .Question, Probability: 0.9974
Token: Questions, Probability: 0.9974
Token: Indicator, Probability: 0.9974

Top 5 tokens for position 18
Token: Question, Probability: 0.9968
Token:  QUESTION, Probability: 0.9968
Token: Questions, Probability: 0.9968
Token: .Question, Probability: 0.9968
Token: Indicator, Probability: 0.9968

Top 5 tokens for position 19
Token: Question, Probability: 0.9970
Token:  QUESTION, Probability: 0.9970
Token: Questions, Probability: 0.9970
Token: .Question, Probability: 0.9970
Token: Indicator, Probability: 0.9970

Top 5 tokens for position 20
Token: Question, Probability: 0.9977
Token:  QUESTION, Probability: 0.9977
Token: Questions, Probability: 0.9977
Token: Indicator, Probability: 0.9977
Token: .Question, Probability: 0.9977

Top 5 tokens for position 21
Token: Question, Probability: 0.9982
Token:  QUESTION, Probability: 0.9982
Token: Questions, Probability: 0.9982
Token: Indicator, Probability: 0.9982
Token: .Question, Probability: 0.9982

Top 5 tokens for position 22
Token: Question, Probability: 0.9983
Token:  QUESTION, Probability: 0.9983
Token: Indicator, Probability: 0.9983
Token: Questions, Probability: 0.9983
Token: .Question, Probability: 0.9983

Top 5 tokens for position 23
Token: Question, Probability: 0.9980
Token:  QUESTION, Probability: 0.9980
Token: Indicator, Probability: 0.9980
Token: Questions, Probability: 0.9980
Token: .Question, Probability: 0.9980

Top 5 tokens for position 24
Token: Question, Probability: 0.9973
Token:  QUESTION, Probability: 0.9973
Token: Indicator, Probability: 0.9973
Token: Questions, Probability: 0.9973
Token: .Question, Probability: 0.9973

Top 5 tokens for position 25
Token: Question, Probability: 0.9971
Token:  QUESTION, Probability: 0.9971
Token: Questions, Probability: 0.9971
Token: Indicator, Probability: 0.9971
Token: .Question, Probability: 0.9971

Top 5 tokens for position 26
Token: Question, Probability: 0.9976
Token:  QUESTION, Probability: 0.9976
Token: Questions, Probability: 0.9976
Token: Indicator, Probability: 0.9976
Token: .Question, Probability: 0.9976

Top 5 tokens for position 27
Token: Question, Probability: 0.9982
Token: Questions, Probability: 0.9982
Token:  QUESTION, Probability: 0.9982
Token: .Question, Probability: 0.9982
Token: Indicator, Probability: 0.9982

Top 5 tokens for position 28
Token: Question, Probability: 0.9984
Token: Questions, Probability: 0.9984
Token:  QUESTION, Probability: 0.9984
Token: .Question, Probability: 0.9984
Token: Indicator, Probability: 0.9984

Top 5 tokens for position 29
Token: Question, Probability: 0.9982
Token: Questions, Probability: 0.9982
Token:  QUESTION, Probability: 0.9982
Token: .Question, Probability: 0.9982
Token: Indicator, Probability: 0.9982

Top 5 tokens for position 30
Token: Question, Probability: 0.9976
Token: Questions, Probability: 0.9976
Token:  QUESTION, Probability: 0.9976
Token: .Question, Probability: 0.9976
Token: Indicator, Probability: 0.9976

Top 5 tokens for position 31
Token: Question, Probability: 0.9974
Token: Questions, Probability: 0.9974
Token:  QUESTION, Probability: 0.9974
Token: .Question, Probability: 0.9974
Token: Indicator, Probability: 0.9974

Top 5 tokens for position 32
Token: Question, Probability: 0.9978
Token: Questions, Probability: 0.9978
Token:  QUESTION, Probability: 0.9978
Token: .Question, Probability: 0.9978
Token: Indicator, Probability: 0.9978

Top 5 tokens for position 33
Token: Question, Probability: 0.9984
Token: Questions, Probability: 0.9984
Token:  QUESTION, Probability: 0.9984
Token: .Question, Probability: 0.9984
Token: Indicator, Probability: 0.9984

Top 5 tokens for position 34
Token: Question, Probability: 0.9986
Token: Questions, Probability: 0.9986
Token: .Question, Probability: 0.9986
Token:  QUESTION, Probability: 0.9986
Token: Indicator, Probability: 0.9986

Top 5 tokens for position 35
Token: Question, Probability: 0.9984
Token: Questions, Probability: 0.9984
Token: .Question, Probability: 0.9984
Token:  QUESTION, Probability: 0.9984
Token: Indicator, Probability: 0.9984

Top 5 tokens for position 36
Token: Question, Probability: 0.9978
Token: Questions, Probability: 0.9978
Token: Indicator, Probability: 0.9978
Token: .Question, Probability: 0.9978
Token:  QUESTION, Probability: 0.9978

Top 5 tokens for position 37
Token: Question, Probability: 0.9970
Token: Questions, Probability: 0.9970
Token:  QUESTION, Probability: 0.9970
Token: Indicator, Probability: 0.9970
Token: .Question, Probability: 0.9970

Top 5 tokens for position 38
Token: Question, Probability: 0.9971
Token: Questions, Probability: 0.9971
Token:  QUESTION, Probability: 0.9971
Token: .Question, Probability: 0.9971
Token: Indicator, Probability: 0.9971

Top 5 tokens for position 39
Token: Question, Probability: 0.9979
Token: Questions, Probability: 0.9979
Token:  QUESTION, Probability: 0.9979
Token: .Question, Probability: 0.9979
Token: Indicator, Probability: 0.9979

Top 5 tokens for position 40
Token: Question, Probability: 0.9986
Token: Questions, Probability: 0.9986
Token:  QUESTION, Probability: 0.9986
Token: .Question, Probability: 0.9986
Token: Indicator, Probability: 0.9986

Top 5 tokens for position 41
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: .Question, Probability: 0.9987
Token:  QUESTION, Probability: 0.9987
Token: Indicator, Probability: 0.9987

Top 5 tokens for position 42
Token: Question, Probability: 0.9984
Token: Questions, Probability: 0.9984
Token: .Question, Probability: 0.9984
Token: Indicator, Probability: 0.9984
Token:  QUESTION, Probability: 0.9984

Top 5 tokens for position 43
Token: Question, Probability: 0.9979
Token: Questions, Probability: 0.9979
Token: .Question, Probability: 0.9979
Token:  QUESTION, Probability: 0.9979
Token: Indicator, Probability: 0.9979

Top 5 tokens for position 44
Token: Question, Probability: 0.9979
Token: Questions, Probability: 0.9979
Token: .Question, Probability: 0.9979
Token:  QUESTION, Probability: 0.9979
Token: Answer, Probability: 0.9979

Top 5 tokens for position 45
Token: Question, Probability: 0.9983
Token: Questions, Probability: 0.9983
Token: .Question, Probability: 0.9983
Token:  QUESTION, Probability: 0.9983
Token: Answer, Probability: 0.9983

Top 5 tokens for position 46
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: .Question, Probability: 0.9988
Token:  QUESTION, Probability: 0.9988
Token: Answer, Probability: 0.9988

Top 5 tokens for position 47
Token: Question, Probability: 0.9989
Token: Questions, Probability: 0.9989
Token: .Question, Probability: 0.9989
Token:  Question, Probability: 0.9989
Token:  QUESTION, Probability: 0.9989

Top 5 tokens for position 48
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: .Question, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: Answer, Probability: 0.9988

Top 5 tokens for position 49
Token: Question, Probability: 0.9985
Token: Questions, Probability: 0.9985
Token: .Question, Probability: 0.9985
Token: Indicator, Probability: 0.9985
Token: Answer, Probability: 0.9985

Top 5 tokens for position 50
Token: Question, Probability: 0.9984
Token: Questions, Probability: 0.9984
Token: .Question, Probability: 0.9984
Token: Answer, Probability: 0.9984
Token: Indicator, Probability: 0.9984

Top 5 tokens for position 51
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: .Question, Probability: 0.9987
Token: Answer, Probability: 0.9987
Token:  Question, Probability: 0.9987

Top 5 tokens for position 52
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: .Question, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: Answer, Probability: 0.9990

Top 5 tokens for position 53
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991
Token: Answer, Probability: 0.9991

Top 5 tokens for position 54
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: .Question, Probability: 0.9990
Token: Answer, Probability: 0.9990

Top 5 tokens for position 55
Token: Question, Probability: 0.9986
Token: Questions, Probability: 0.9986
Token: .Question, Probability: 0.9986
Token: Answer, Probability: 0.9986
Token:  Question, Probability: 0.9986

Top 5 tokens for position 56
Token: Question, Probability: 0.9982
Token: Questions, Probability: 0.9982
Token: Answer, Probability: 0.9982
Token: .Question, Probability: 0.9982
Token: Indicator, Probability: 0.9982

Top 5 tokens for position 57
Token: Question, Probability: 0.9983
Token: Questions, Probability: 0.9983
Token: Answer, Probability: 0.9983
Token: .Question, Probability: 0.9983
Token:  QUESTION, Probability: 0.9983

Top 5 tokens for position 58
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: .Question, Probability: 0.9988

Top 5 tokens for position 59
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token: .Question, Probability: 0.9990

Top 5 tokens for position 60
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 61
Token: Question, Probability: 0.9989
Token: Questions, Probability: 0.9989
Token: Answer, Probability: 0.9989
Token:  Question, Probability: 0.9989
Token: .Question, Probability: 0.9989

Top 5 tokens for position 62
Token: Question, Probability: 0.9986
Token: Questions, Probability: 0.9986
Token: Answer, Probability: 0.9986
Token: .Question, Probability: 0.9986
Token:  Question, Probability: 0.9986

Top 5 tokens for position 63
Token: Question, Probability: 0.9985
Token: Questions, Probability: 0.9985
Token: Answer, Probability: 0.9985
Token: .Question, Probability: 0.9985
Token:  Question, Probability: 0.9985

Top 5 tokens for position 64
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: .Question, Probability: 0.9988

Top 5 tokens for position 65
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: .Question, Probability: 0.9990

Top 5 tokens for position 66
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 67
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: .Question, Probability: 0.9990

Top 5 tokens for position 68
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: .Question, Probability: 0.9988

Top 5 tokens for position 69
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: Answer, Probability: 0.9987
Token:  Question, Probability: 0.9987
Token: .Question, Probability: 0.9987

Top 5 tokens for position 70
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: .Question, Probability: 0.9990

Top 5 tokens for position 71
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token: .Question, Probability: 0.9992

Top 5 tokens for position 72
Token: Question, Probability: 0.9993
Token: Questions, Probability: 0.9993
Token:  Question, Probability: 0.9993
Token: Answer, Probability: 0.9993
Token: .Question, Probability: 0.9993

Top 5 tokens for position 73
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token: .Question, Probability: 0.9992

Top 5 tokens for position 74
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: .Question, Probability: 0.9990

Top 5 tokens for position 75
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: .Question, Probability: 0.9988

Top 5 tokens for position 76
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: .Question, Probability: 0.9988

Top 5 tokens for position 77
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 78
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token: .Question, Probability: 0.9992

Top 5 tokens for position 79
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token: .Question, Probability: 0.9992

Top 5 tokens for position 80
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 81
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: Option, Probability: 0.9988

Top 5 tokens for position 82
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: Answer, Probability: 0.9987
Token:  Question, Probability: 0.9987
Token: Option, Probability: 0.9987

Top 5 tokens for position 83
Token: Question, Probability: 0.9989
Token: Questions, Probability: 0.9989
Token: Answer, Probability: 0.9989
Token:  Question, Probability: 0.9989
Token: .Question, Probability: 0.9989

Top 5 tokens for position 84
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 85
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 86
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: .Question, Probability: 0.9990

Top 5 tokens for position 87
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: Answer, Probability: 0.9987
Token:  Question, Probability: 0.9987
Token: Option, Probability: 0.9987

Top 5 tokens for position 88
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: Answer, Probability: 0.9987
Token:  Question, Probability: 0.9987
Token: Option, Probability: 0.9987

Top 5 tokens for position 89
Token: Question, Probability: 0.9990
Token: Questions, Probability: 0.9990
Token: Answer, Probability: 0.9990
Token:  Question, Probability: 0.9990
Token: Option, Probability: 0.9990

Top 5 tokens for position 90
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: .Question, Probability: 0.9992

Top 5 tokens for position 91
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: .Question, Probability: 0.9992

Top 5 tokens for position 92
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: .Question, Probability: 0.9991

Top 5 tokens for position 93
Token: Question, Probability: 0.9989
Token: Questions, Probability: 0.9989
Token: Answer, Probability: 0.9989
Token:  Question, Probability: 0.9989
Token: Option, Probability: 0.9989

Top 5 tokens for position 94
Token: Question, Probability: 0.9987
Token: Questions, Probability: 0.9987
Token: Answer, Probability: 0.9987
Token:  Question, Probability: 0.9987
Token: Option, Probability: 0.9987

Top 5 tokens for position 95
Token: Question, Probability: 0.9988
Token: Questions, Probability: 0.9988
Token: Answer, Probability: 0.9988
Token:  Question, Probability: 0.9988
Token: Option, Probability: 0.9988

Top 5 tokens for position 96
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: Option, Probability: 0.9991

Top 5 tokens for position 97
Token: Question, Probability: 0.9993
Token: Questions, Probability: 0.9993
Token:  Question, Probability: 0.9993
Token: Answer, Probability: 0.9993
Token: Option, Probability: 0.9993

Top 5 tokens for position 98
Token: Question, Probability: 0.9992
Token: Questions, Probability: 0.9992
Token:  Question, Probability: 0.9992
Token: Answer, Probability: 0.9992
Token: Option, Probability: 0.9992

Top 5 tokens for position 99
Token: Question, Probability: 0.9991
Token: Questions, Probability: 0.9991
Token: Answer, Probability: 0.9991
Token:  Question, Probability: 0.9991
Token: Option, Probability: 0.9991

Transcription Results:
--------------------------------------------------
Model output: QuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestionQuestion
Reference  : IT IS YOU WHO ARE MISTAKEN RAOUL I HAVE READ HIS DISTRESS IN HIS EYES IN HIS EVERY GESTURE AND ACTION THE WHOLE DAY
--------------------------------------------------
```
