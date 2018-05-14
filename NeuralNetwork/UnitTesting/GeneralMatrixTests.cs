using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuronalNetwork;
using NUnit.Framework;

namespace NeuralNetwork.UnitTesting
{
    [TestFixture]
    class GeneralMatrixTests
    {
        [Test]
        public void MatrixTranspose()
        {
            Matrix m1 = new Matrix(3, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = -65;
            m1.Value[1][1] = 11;
            m1.Value[2][0] = -32;
            m1.Value[2][1] = 1;

            var res = Matrix.Transpose(m1);

            Assert.That(m1.Rows == res.Columns && m1.Columns == res.Rows);
            
            Assert.That(Math.Abs(res.Value[0][0] - 13) < 0.01);
            Assert.That(Math.Abs(res.Value[0][1] - (-65)) < 0.01);
            Assert.That(Math.Abs(res.Value[0][2] - (-32)) < 0.01);
            Assert.That(Math.Abs(res.Value[1][0] - 32) < 0.01);
            Assert.That(Math.Abs(res.Value[1][1] - 11) < 0.01);
            Assert.That(Math.Abs(res.Value[1][2] - 1) < 0.01);
        }

        [Test]
        public void MatrixMap()
        {
            Matrix m1 = new Matrix(2, 2);

            m1.Value[0][0] = 0.3;
            m1.Value[0][1] = 1;
            m1.Value[1][0] = -0.1;
            m1.Value[1][1] = 0.7;

            var sigmoid = new Func<double, float>(y =>
            {
                float k = (float)Math.Exp(y);
                return k / (1.0f + k);
            });

            m1.Map(sigmoid);
            
            Assert.That(Math.Abs(m1.Value[0][0] - 0.5744) < 0.01);
            Assert.That(Math.Abs(m1.Value[0][1] - 0.731) < 0.01);
            Assert.That(Math.Abs(m1.Value[1][0] - 0.475) < 0.01);
            Assert.That(Math.Abs(m1.Value[1][1] - 0.668) < 0.01);
        }
    }
}
