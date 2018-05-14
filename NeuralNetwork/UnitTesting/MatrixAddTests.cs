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
    class MatrixAddTests
    {
        [Test]
        public void MatrixAddTest2By2With2By2()
        {
            Matrix m1 = new Matrix(2, 2);
            Matrix m2 = new Matrix(2, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = -65;
            m1.Value[1][1] = 11;

            m2.Value[0][0] = 3;
            m2.Value[0][1] = 6421;
            m2.Value[1][0] = 75;
            m2.Value[1][1] = -11;

            m1.Add(m2);

            Assert.That(m1.Rows == 2 && m1.Columns == 2);

            Assert.That(m1.Value[0][0] == 16);
            Assert.That(m1.Value[0][1] == 6453);
            Assert.That(m1.Value[1][0] == 10);
            Assert.That(m1.Value[1][1] == 0);
        }

        [Test]
        public void MatrixAddTest2By2With1()
        {
            Matrix m1 = new Matrix(2, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = -65;
            m1.Value[1][1] = 11;

            m1.Add(-5);

            Assert.That(m1.Rows == 2 && m1.Columns == 2);

            Assert.That(Math.Abs(m1.Value[0][0] - 8) < 0.01);
            Assert.That(Math.Abs(m1.Value[0][1] - 27) < 0.01);
            Assert.That(Math.Abs(m1.Value[1][0] - (-70)) < 0.01);
            Assert.That(Math.Abs(m1.Value[1][1] - 6) < 0.01);
        }
    }
}
