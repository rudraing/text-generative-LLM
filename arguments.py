import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='This is a demonstration program')
    parser.add_argument('-llms', type=str, required=True, help='Please provide an llms')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f'The provided llms is: {args.llms}')

if __name__ == '__main__':
    main()
