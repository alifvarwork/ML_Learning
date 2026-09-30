using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using ML;
using ML.Regression;

ModelisMathsFormula();

static void SimplestMLCode()
{
    var mlcontext = new MLContext();
    var data = mlcontext.Data.LoadFromEnumerable<InsuranceData>(DataRegression.GetInsuranceData());
    //pipeline features, label, transformers, trainers
    var pipline = mlcontext.Transforms.Concatenate("Features", ["Age"])
        .Append(mlcontext.Regression.Trainers.Ols(labelColumnName: "Premium", featureColumnName: "Features"));

    var model = pipline.Fit(data);

    var predictionEngine = mlcontext.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);

    var result = predictionEngine.Predict(new InsuranceData { Age = 85 });//Inference

    Console.WriteLine(result.PredictedPremium);
}

static void ModelisMathsFormula()
{
    var mlcontext = new MLContext();
    var data = mlcontext.Data.LoadFromEnumerable<InsuranceData>(DataRegression.GetInsuranceData());
    //pipeline features, label, transformers, trainers
    var pipline = mlcontext.Transforms.Concatenate("Features", ["Age"])
        .Append(mlcontext.Regression.Trainers.Ols(labelColumnName: "Premium", featureColumnName: "Features"));

    var model = pipline.Fit(data);

    var pipelineSteps = model as IEnumerable<ITransformer>;

    var modelLastStep = pipelineSteps.Last() as RegressionPredictionTransformer<OlsModelParameters>;
    var modelParameters = modelLastStep.Model;
    float intercept = modelParameters.Bias;
    var slope = modelParameters.Weights[0];

    Console.WriteLine("Model is a linear regression model with the following parameters:");
    Console.WriteLine($"Intercept: {intercept}, Slope: {slope}");
    Console.WriteLine($"Equation: y = {slope}x + {intercept}"); 
    Console.ReadLine();
}