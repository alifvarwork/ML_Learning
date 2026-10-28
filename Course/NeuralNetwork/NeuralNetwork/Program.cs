NeuralNetwork neuralNetwork = new NeuralNetwork();

double[,] trainingSetInputs = new double[,]
{
    { 0, 0, 0},
    { 1, 1, 1 },
    { 1, 0, 0 },
};

double[,] trainingSetOutputs = new double[,]
{
    { 0 },
    { 1 },
    { 1 },
};

neuralNetwork.Train(trainingSetInputs, trainingSetOutputs, 1000);

double[,] output = neuralNetwork.Predict(new double[,] { { 0, 1, 0}, { 0, 0, 0}, { 0, 0, 1}, { 1, 0, 0 } });

PrintMatrix(output);

static void PrintMatrix(double[,] matrix)
{
    int numberOfRows = matrix.GetLength(0);
    int numberOfColumns = matrix.GetLength(1);
    for (int i = 0; i < numberOfRows; i++)
    {
        for (int j = 0; j < numberOfColumns; j++)
        {
            Console.Write(Math.Round(matrix[i, j]) + " ");
        }
        Console.WriteLine();
    }
}