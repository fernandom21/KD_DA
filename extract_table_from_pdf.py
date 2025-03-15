import os
import argparse

import pandas as pd
from pypdf import PdfReader
#from pdfminer import high_level


def read_pdf_page(fp, page_number, output_file, pdfminer=False):
    if pdfminer:
        text = high_level.extract_text(fp, "", [page_number])
    else:
        reader = PdfReader(fp)

        print(len(reader.pages))
        page = reader.pages[page_number]

        text = page.extract_text()

    print(text)

    print(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
       f.write(text)

    return text


def save_pdf_as_csv(text, output_file):
    lines = text.split('\n')

    table_data = [line.split(' ') for line in lines]

    df = pd.DataFrame(table_data)
    df.to_csv(output_file, index=False, header=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('papers', 's3mix.pdf'),
                        help='paper pdf title')
    parser.add_argument('--pdf_page_number', type=int, default=8,
                        help='pdf page number starting from 0')
    parser.add_argument('--pdfminer', action='store_true')

    # output
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'extracted_text'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    fn = os.path.splitext(os.path.split(args.input_file)[-1])[0]
    output_file = f'{fn}_{args.pdf_page_number}'
    args.output_file = os.path.join(args.results_dir, output_file)

    text = read_pdf_page(args.input_file, args.pdf_page_number,
                         f'{args.output_file}.txt', args.pdfminer)

    save_pdf_as_csv(text, f'{args.output_file}.csv')

    return 0


if __name__ == '__main__':
    main()
  

'''
table = "baseline 82.45 84.86 85.72 86.24 91.51 92.33 92.94 93.06 88.27 89.53 90.31 90.46
Mixup [45] 83.21 85.23 86.26 87.42 91.56 92.65 92.74 93.05 87.82 89.77 90.10 90.31
CutMix [42] 83.40 85.81 86.78 87.50 92.24 93.28 93.42 93.25 88.78 90.19 91.12 91.18
Cutout [5] 81.81 84.69 84.43 86.12 91.46 92.65 92.80 93.02 88.03 89.35 89.62 90.07
InPS [46] 84.55 86.07 86.83 87.57 91.98 92.08 93.26 92.66 89.08 90.16 91.48 90.46
SnapMix [17] 83.57 86.40 87.02 87.54 92.24 93.22 93.22 93.51 89.32 90.43 91.27 91.60
S3Mix 85.14 86.68 87.50 87.71 92.63 93.27 93.61 93.65 89.53 90.46 91.51 91.54
S3Mix† 85.43 86.83 87.52 87.97 92.95 93.31 93.75 93.92 89.80 90.76 91.60 91.81"
table = """baseline 82.45 84.86 85.72 86.24 91.51 92.33 92.94 93.06 88.27 89.53 90.31 90.46
Mixup [45] 83.21 85.23 86.26 87.42 91.56 92.65 92.74 93.05 87.82 89.77 90.10 90.31
CutMix [42] 83.40 85.81 86.78 87.50 92.24 93.28 93.42 93.25 88.78 90.19 91.12 91.18
Cutout [5] 81.81 84.69 84.43 86.12 91.46 92.65 92.80 93.02 88.03 89.35 89.62 90.07
InPS [46] 84.55 86.07 86.83 87.57 91.98 92.08 93.26 92.66 89.08 90.16 91.48 90.46
SnapMix [17] 83.57 86.40 87.02 87.54 92.24 93.22 93.22 93.51 89.32 90.43 91.27 91.60
S3Mix 85.14 86.68 87.50 87.71 92.63 93.27 93.61 93.65 89.53 90.46 91.51 91.54
S3Mix† 85.43 86.83 87.52 87.97 92.95 93.31 93.75 93.92 89.80 90.76 91.60 91.81"""
table

'''
