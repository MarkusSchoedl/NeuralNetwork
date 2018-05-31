using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuronalNetwork
{
    class NeuronalNetwork
    {
        private readonly Matrix _weigths_ih;
        private readonly Matrix _weigths_ho;

        private readonly Matrix _biasH;
        private readonly Matrix _biasO;
        private const double LearningRate = 0.18;

        private Matrix output;

        public NeuronalNetwork(int input, int hidden, int output)
        {
            _weigths_ih = new Matrix(hidden, input);
            _weigths_ho = new Matrix(output, hidden);
            _weigths_ih.Randomize();
            _weigths_ho.Randomize();

            _biasH = new Matrix(hidden, 1);
            _biasO = new Matrix(output, 1);
            _biasH.Randomize();
            _biasO.Randomize();
        }

        public int FeedForward(double[] inputArray)
        {
            //Generating the hidden outputs
            var inputs = Matrix.FromArray(inputArray);
            var hidden = Matrix.Multiply(_weigths_ih, inputs);
            hidden.Add(_biasH);
            //activation function
            hidden.Map(Sigmoid);

            output = Matrix.Multiply(_weigths_ho, hidden);
            output.Add(_biasO);
            output.Map(Sigmoid);

            var outArr = output.ToArray();
            return Array.IndexOf(outArr, outArr.Max());
        }

        public static float Sigmoid(double value)
        {
            float k = (float)Math.Exp(value);
            float res = k / (1.0f + k);

            if (float.IsNaN(res))
            {
                return k < 0 ? -1 : 1;
            }

            return res;
        }

        public static float dSigmoid(double value)
        {
            return (float)(value * (1 - value));
        }

        public int Train(double[] inputArray, double[] targetArray)
        {
            //Generating the hidden outputs
            var inputs = Matrix.FromArray(inputArray);
            var hidden = Matrix.Multiply(_weigths_ih, inputs);
            hidden.Add(_biasH);

            //activation function
            hidden.Map(Sigmoid);

            output = Matrix.Multiply(_weigths_ho, hidden);
            output.Add(_biasO);
            output.Map(Sigmoid);

            var targets = Matrix.FromArray(targetArray);
            // Calculate the error
            // ERROR = TARGETS - OUTPUTS
            var outputErrors = Matrix.Substract(targets, output);

            // gradient = outputs * (1 - outputs);
            // Calculate gradient
            var gradients = Matrix.Map(output, dSigmoid);
            gradients.Multiply(outputErrors);
            gradients.Multiply(LearningRate);

            // Calculate deltas
            var hidden_T = Matrix.Transpose(hidden);
            var weight_ho_deltas = Matrix.Multiply(gradients, hidden_T);

            //Adjust the weights by deltas
            _weigths_ho.Add(weight_ho_deltas);
            //Adjust the bias by its deltas
            _biasO.Add(gradients);

            //Calculate the hidden layer errors
            var w_ho_t = Matrix.Transpose(_weigths_ho);
            var hiddenErrors = Matrix.Multiply(w_ho_t, outputErrors);

            //Calculate hidden gradient
            var hidden_gradient = Matrix.Map(hidden, dSigmoid);
            hidden_gradient.Multiply(hiddenErrors);
            hidden_gradient.Multiply(LearningRate);

            //Calculate inputArray-hidden deltas
            var inputs_T = Matrix.Transpose(inputs);
            var weight_ih_deltas = Matrix.Multiply(hidden_gradient, inputs_T);
            _weigths_ih.Add(weight_ih_deltas);
            //Adjust the bias by its deltas
            _biasH.Add(hidden_gradient);

            var outArr = output.ToArray();
            return Array.IndexOf(outArr, outArr.Max());
        }

        public double CalculateEpsilon(int label)
        {
            if (output == null)
            {
                throw new Exception("No calculation happened");
            }

            double error = 0.0;

            for (int i = 0; i < output.Rows; i++)
            {
                error += Math.Pow(output.Value[i][0] - (label == i ? 1 : 0), 2);
            }
            error = error / output.Rows;

            return error;
        }
    }
}