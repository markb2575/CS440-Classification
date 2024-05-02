import digits
import face

#Trains all models
face.training()
digits.training()

#Run specific test cases
# face.runTestNeural(15)
# face.runTestPerceptron(15)
# digits.runTestNeural(15)
# digits.runTestPerceptron(15)