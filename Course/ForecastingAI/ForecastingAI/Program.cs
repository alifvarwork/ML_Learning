using ForecastingAI;
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext();
var dataView = mlContext.Data.LoadFromTextFile<StockData>("stock_data.Csv", separatorChar: ',');
var preview = dataView.Preview();
foreach (var row in preview.RowView)
{
    Console.WriteLine($"{row.Values[0]} | {row.Values[1]}");
}

var pipeline = mlContext.Transforms.Concatenate("Features", nameof(StockData.Open),  nameof(StockData.High), nameof(StockData.Low)).Append(mlContext.Transforms.CopyColumns("Label", nameof(StockData.Close)))
    .Append(mlContext.Regression.Trainers.Ols());
var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
var model = pipeline.Fit(trainTestData.TrainSet);
var predictions = model.Transform(trainTestData.TestSet);
var metrics = mlContext.Regression.Evaluate(predictions);
Console.WriteLine($"R-Squared: {metrics.RSquared}");
Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

var predictionResults = mlContext.Data.CreateEnumerable<StockPrediction>(predictions, reuseRowObject: false);
var testData = mlContext.Data.CreateEnumerable<StockData>(trainTestData.TestSet, reuseRowObject: false);
foreach (var (predicted, actual) in predictionResults.Zip(testData, (p, a) => (p, a)))
{
    Console.WriteLine($"Actual Close Price: {actual.Close}, Predicted Close Price:{predicted.PredictedClose}");
}