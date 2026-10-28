public class NeuralNetwork
{
    private double[,] weights;
    enum Operation { Multiply, Add, Subtract }

    public NeuralNetwork()
    {
        Random random = new Random(1);
        int numberOfInputNodes = 3;
        int numberOfOutputNodes = 1;
        weights = new double[numberOfInputNodes, numberOfOutputNodes];
        for (int i = 0; i < numberOfInputNodes; i++)
        {
            for (int j = 0; j < numberOfOutputNodes; j++)
            {
                weights[i, j] = 2 * random.NextDouble() - 1;
            }
        }
    }

    private double[,] Activate(double[,] matrix, bool isDerivative)
    {
        int numberOfRows = matrix.GetLength(0);
        int numberOfColumns = matrix.GetLength(1);

        double[,] result = new double[numberOfRows, numberOfColumns];
        for (int i = 0; i < numberOfRows; i++)
        {
            for (int j = 0; j < numberOfColumns; j++)
            {
                if (isDerivative)
                {
                    result[i, j] = matrix[i, j] * (1 - matrix[i, j]);
                }
                else
                {
                    result[i, j] = 1 / (1 + Math.Exp(-matrix[i, j]));
                }
            }
        }

        // Implementation for activation function
        return result;
    }

    public void Train(double[,] trainingInputs, double[,] trainingOutputs, int iterations)
    {
        for (int i = 0; i < iterations; i++)
        {
            double[,] predictedOutput = Predict(trainingInputs);
            double[,] error = PerformOperation(trainingOutputs, predictedOutput, Operation.Subtract);
            double[,] adjustments = DotProduct(Transpose(trainingInputs), PerformOperation(error, Activate(predictedOutput, true), Operation.Multiply));
            weights = PerformOperation(weights, adjustments, Operation.Add);
        }
    }

    private double[,] PerformOperation(double[,] matrixA, double[,] matrixB, Operation operation)
    {
        int numberOfRows = matrixA.GetLength(0);
        int numberOfColumns = matrixA.GetLength(1);
        double[,] result = new double[numberOfRows, numberOfColumns];
        for (int i = 0; i < numberOfRows; i++)
        {
            for (int j = 0; j < numberOfColumns; j++)
            {
                switch (operation)
                {
                    case Operation.Multiply:
                        result[i, j] = matrixA[i, j] * matrixB[i, j];
                        break;
                    case Operation.Add:
                        result[i, j] = matrixA[i, j] + matrixB[i, j];
                        break;
                    case Operation.Subtract:
                        result[i, j] = matrixA[i, j] - matrixB[i, j];
                        break;
                }
            }
        }
        return result;
    }

    private double[,] DotProduct(double[,] matrixA, double[,] matrixB)
    {
        int numberOfRowsA = matrixA.GetLength(0);
        int numberOfColumnsA = matrixA.GetLength(1);
        int numberOfRowsB = matrixB.GetLength(0);
        int numberOfColumnsB = matrixB.GetLength(1);
        if (numberOfColumnsA != numberOfRowsB)
        {
            throw new InvalidOperationException("Matrix dimensions are not compatible for dot product.");
        }
        double[,] result = new double[numberOfRowsA, numberOfColumnsB];
        for (int i = 0; i < numberOfRowsA; i++)
        {
            for (int j = 0; j < numberOfColumnsB; j++)
            {
                double sum = 0;
                for (int k = 0; k < numberOfColumnsA; k++)
                {
                    sum += matrixA[i, k] * matrixB[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }


    private double[,] Transpose(double[,] matrix)
    {
        return matrix.Cast<double>().ToArray().Transpose(matrix.GetLength(0), matrix.GetLength(1));
    }

    public double[,] Predict(double[,] inputs)
    {
        double[,] output = DotProduct(inputs, weights);
        return Activate(output, false);
    }


}

