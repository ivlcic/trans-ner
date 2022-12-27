#!/usr/bin/env python
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='JSON to CSV convert')
    parser.add_argument('file_path', help='Path to json file')
    args = parser.parse_args()
    csv = 'lang\tmodel\tPER p\tPER r\tPER f1\t PER n'
    csv += '\tLOC p\tLOC r\tLOC f1\t LOC n'
    csv += '\tORG p\tORG r\tORG f1\t ORG n'
    csv += '\tTotal p\tTotal r\tTotal f1\t Total acc\n'
    with open(args.file_path, 'rt', encoding='utf-8') as jfp:
        data = json.load(jfp)
        for test_result in data:
            test_name = list(test_result.keys())[0]
            test_names = test_name.split('-')
            test_data = test_result[test_name]
            csv += test_names[0].strip() + '\t'
            csv += test_names[1].strip() + '\t'
            csv += str(test_data['PER']['precision']) + '\t'
            csv += str(test_data['PER']['recall']) + '\t'
            csv += str(test_data['PER']['f1']) + '\t'
            csv += str(test_data['PER']['number']) + '\t'
            csv += str(test_data['LOC']['precision']) + '\t'
            csv += str(test_data['LOC']['recall']) + '\t'
            csv += str(test_data['LOC']['f1']) + '\t'
            csv += str(test_data['LOC']['number']) + '\t'
            csv += str(test_data['ORG']['precision']) + '\t'
            csv += str(test_data['ORG']['recall']) + '\t'
            csv += str(test_data['ORG']['f1']) + '\t'
            csv += str(test_data['ORG']['number']) + '\t'
            csv += str(test_data['overall_precision']) + '\t'
            csv += str(test_data['overall_recall']) + '\t'
            csv += str(test_data['overall_f1']) + '\t'
            csv += str(test_data['overall_accuracy']) + '\n'

    with open(args.file_path + '.csv', 'wt', encoding='utf-8') as fp:
        fp.write(csv)
