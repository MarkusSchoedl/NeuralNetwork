using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using static NeuronalNetwork.GetImages;

namespace NeuronalNetwork
{
    class Program
    {
        private double[,] data = new double[4, 2] {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

        static void Main(string[] args)
        {
            //string location = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            //var images = GetImagesFromFile(location);

            var nn = new NeuronalNetwork(2, 2, 1);

            Random gen = new Random();

            for (int i = 0; i < 300000; i++)
            {
                double val1 = (int)(gen.NextDouble() + 0.5);
                double val2 = (int)(gen.NextDouble() + 0.5);

                double[] input = { val1, val2 };
                double[] target;

                if (val1 == 0 && val2 == 1 || val1 == 1 && val2 == 0)
                {
                    target = new[] { 1.0 };
                }
                else
                {
                    target = new[] { 0.0 };
                }

                var output = nn.Train(input, target);

                if (i % 100 == 0)
                {
                    Console.WriteLine("input: " + (int)val1 + "  " + (int)val2 + "  expected: " +
                                      (int)(double)target[0]);
                    Console.WriteLine(output.Value[0][0]);
                }

                //if (Console.ReadKey().KeyChar == 'Q')
                //    break;
            }

            Console.WriteLine("0 0 - " + nn.FeedForward(new double[] { 0.0, 0.0 })[0]);
            Console.WriteLine("1 0 - " + nn.FeedForward(new double[] { (double)0.0, (double)1.0 })[0]);
            Console.WriteLine("0 1 - " + nn.FeedForward(new double[] { (double)1.0, 0.0 })[0]);
            Console.WriteLine("1 1 - " + nn.FeedForward(new double[] { (double)1.0, 1.0 })[0]);

            Console.ReadKey();
        }
    }
}
