public static class Extenstions
{
    public static double[,] Transpose(this double[] array, int rows, int columns)
    {
        double[,] transposedMatrix = new double[columns, rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                transposedMatrix[j, i] = array[i * columns + j];
            }
        }
        return transposedMatrix;
    }
}

