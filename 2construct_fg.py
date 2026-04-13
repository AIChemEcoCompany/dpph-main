import os,sys
import argparse
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from utils.connect_fg import construct_fg_main12,main3,main4



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description = 'construction of FG',
        epilog = 'example 2construct_fg.py --level 1'
    )
    
    parser.add_argument(
        '--level',
        type = int,
        choices = [1, 2, 3, 4],  
        default = 4,  
        help='Which Level of broken/formed bond (values are 1, 2, 3, 4, default is 4)'
    )

    arg = parser.parse_args()
    n = arg.level
    

    print(f'Level {n} generating...')
    if n == 1 or n == 2:
        construct_fg_main12(n)
    elif n ==3 :
        main3()
    elif n == 4:
        main4()





