using System;
using System.Collections.Generic;
using System.Text;

namespace NeuronalNetwork
{

    public class DigitImage
    {
        private readonly int _label;
        private readonly double[] _data;

        public DigitImage(byte[][] pixels, byte label)
        {
            _data = new double[28 * 28];

            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    _data[i * 28 + j] = pixels[i][j];
                }
            }

            Otsu();

            _label = label;
        }

        public override string ToString()
        {
            string sb = _label + " = " + _data.Length + ": (\n";

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    if ((int)_data[(y * 28) + x] == 1)
                    {
                        sb += "*";
                    }
                    else
                    {
                        sb += " ";
                    }

                }
                sb += "\n";
            }
            sb += ")";

            return sb;
        }

        //Uses Otsu's Threshold algorithm to convert from grayscale to black and white
        private void Otsu()
        {
            int[] histogram = new int[256];

            foreach (double datum in _data)
            {
                histogram[(int)datum]++;
            }

            double sum = 0;
            for (int j = 0; j < histogram.Length; j++)
            {
                sum += j * histogram[j];
            }

            double sumB = 0;
            int wB = 0;
            int wF;

            double maxVariance = 0;
            int threshold = 0;

            int i = 0;
            bool found = false;

            while (i < histogram.Length && !found)
            {
                wB += histogram[i];

                if (wB != 0)
                {
                    wF = _data.Length - wB;

                    if (wF != 0)
                    {
                        sumB += (i * histogram[i]);

                        double mB = sumB / wB;
                        double mF = (sum - sumB) / wF;

                        double varianceBetween = wB * Math.Pow((mB - mF), 2);

                        if (varianceBetween > maxVariance)
                        {
                            maxVariance = varianceBetween;
                            threshold = i;
                        }
                    }
                    else
                    {
                        found = true;
                    }
                }

                i++;
            }

            for (i = 0; i < _data.Length; i++)
            {
                _data[i] = _data[i] <= threshold ? 0 : 1;
            }
        }

        public int GetLabel()
        {
            return _label;
        }

        public double[] GetData()
        {
            return _data;
        }
    }
}
