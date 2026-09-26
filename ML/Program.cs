using Microsoft.ML;
using ML;
using ML.Regression;

var mlcontext = new MLContext();
var data = mlcontext.Data.LoadFromEnumerable<InsuranceData>(DataRegression.GetInsuranceData());
//pipeline features, label, transformers, trainers
var pipline = mlcontext.Transforms.Concatenate("Features", ["Age"])
    .Append(mlcontext.Regression.Trainers.Ols( labelColumnName: "Premium", featureColumnName: "Features"));

var model = pipline.Fit(data);

var predictionEngine = mlcontext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);

var result = predictionEngine.Predict(new InsuranceData { Age = 85 });

Console.WriteLine(result.PredictedPremium);