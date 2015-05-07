#!/usr/bin/env python
import sys

def main():
    if len(sys.argv) < 8 or "-h" in sys.argv or "--help" in sys.argv:
        print_usage()
        sys.exit()

    #sys.argv = [sys.argv[0]] + [.81, 10, -10, -5, -1, 3, 1]
    given_available, r_park, r_crash, r_handicap, r_drive, spots, r_distance  = [float(arg) for arg in sys.argv[1:]]
    p_handicap = .9
    spots = int(spots)
    print 8*int(spots), 2
    print ""
    """
    generates movement probabilities for each state
    
    our states are this:
    Row, Available (t/f), Parked (t/f)
    RowRowB Unavailable RowA Available RowB Available
   
    F,F, T,F F,T, T,T 

    A1 A2 A3 B1 B2 B3, A1 A2 A3 B1 B2 B3, ....
    """
    # 2 rows and 4 distinct states
    for i in range(spots * 2 * 4):
        for j in range(spots * 2 * 4):
            i_spot_num = i / 4 #( 0 .. spots * 2)
            j_spot_num = j / 4
            # could probably write functions to make this code look better
            # Handles wrap around state
            if i_spot_num + 1 == (spots) * 2:
                next_store_distance = spots
            elif i_spot_num + 1 < (spots):
                next_store_distance = spots - (i_spot_num + 1)
            else:
                next_store_distance = (i_spot_num + 2) - spots


            p_available = given_available * next_store_distance / (spots + 1)
            # i % 4 represents our current state space
            # j % 4 represents the next state space

            if j_spot_num == spots - 1 and j_spot_num == i_spot_num + 1 and next_store_distance == 1:
                if j % 4 == 0 and i % 4 < 2:
                    print "%0.2f" % (1 - p_handicap),
                elif j % 4 == 1 and i % 4 < 2:
                    print "%0.2f" % p_handicap,
                #refer to park action
                elif i == j:
                    print "%0.2f" % 1.00,
                else:
                    print "%0.2f" % 0.00,

            elif i_spot_num + 1 == spots and j_spot_num == spots and next_store_distance == 1:
                if j % 4 == 0 and i % 4 < 2:
                    print "%0.2f" % (1 - p_handicap),
                elif j % 4 == 1 and i % 4 < 2:
                    print "%0.2f" % p_handicap,
                #refer to park action
                elif i == j:
                    print "%0.2f" % 1.00,
                else:
                    print "%0.2f" % 0.00,

            elif j_spot_num == i_spot_num + 1:
                #not empty and not parked
                if j % 4 == 0 and i % 4 < 2:
                    print "%0.2f" % (1 - p_available),
                #empty and not parked
                elif j % 4 == 1 and i % 4 < 2:
                    print "%0.2f" % p_available,
                #refer to park action
                elif i == j:
                    print "%0.2f" % 1.00,
                else:
                    print "%0.2f" % 0.00,
            elif j_spot_num == 0 and i_spot_num == spots * 2 - 1:
                #not empty and not parked
                if j % 4 == 0 and i % 4 < 2:
                    print "%0.2f" % (1 - p_available),
                #empty and not parked
                elif j % 4 == 1 and i % 4 < 2:
                    print "%0.2f" % p_available,
                #refer to park action
                elif i == j:
                    print "%0.2f" % 1.00,
                else:
                    print "%0.2f" % 0.00,
            elif i == j and i % 4 > 1:
                    print "%0.2f" % 1.00,
            else:
                print "%0.2f" % 0.00,
        print ""
    print ""

    #print park action
    for i in range(spots * 2 * 4):
        for j in range(spots * 2 * 4):
            i_spot_num = i / 4 #( 0 .. spots * 2)
            j_spot_num = j / 4
            #if youre in an available space, parking takes you to parked and available at the same spot
            if i % 4 == 1 and j % 4 == 3 and j_spot_num == i_spot_num:
                print "%0.2f" % 1.0,
            #if youre in an unavailable space, parking taks you to parked and unavailable space
            elif i % 4 == 0 and j % 4 == 2 and j_spot_num == i_spot_num:
                print "%0.2f" % 1.0,
            #if youre in a parked state you stay where you are
            elif i % 4 > 1 and i == j:
                print "%0.2f" % 1.0,
            else:
                print "%0.2f" % 0.0,
        print ""
    print ""


    #print rewards
    for i in range(spots * 4 * 2):
        i_spot_num = i / 4 #( 0 .. spots * 2)
        # could probably write functions to make this code look better
        # Handles wrap around state
        if i_spot_num  == (spots) * 2:
            store_distance = spots
        elif i_spot_num < (spots):
            store_distance = spots - (i_spot_num )
        else:
            store_distance = (i_spot_num + 1) - spots
        #handicap
        if store_distance == 1:
            #driving
            if i % 4 < 2:
                print "%0.2f" % r_drive,
            #parked and occupado
            if i % 4 == 2:
                print "%0.2f" % r_crash,
            #parked success
            if i % 4 == 3:
                print "%0.2f" % (r_handicap + r_distance * store_distance),
        else:
            #driving
            if i % 4 < 2:
                print "%0.2f" % r_drive,
            #parked and occupied
            if i % 4 == 2:
                print "%0.2f" % r_crash,
            #parked success
            if i % 4 == 3:
                print "%0.2f" % (r_park + r_distance * store_distance),

                
                 
    

def print_usage():
    print >> sys.stderr, "Creates a parking MDP based on specified parameters, A1 and B1 are handicap spots, probability of availble is a linear function of distance, p * distance/nspots. Rewards are applied, may specify negative. distance reward is applied for each tile the person needs to walk" 
    print >> sys.stderr, "create_parking_mdp.py\t<p_available>\t<parking_reward>\t<crashing_reward>\t<handicap_reward>\t<drive_reward>\t<n_spots>\t<distance_reward> "

if __name__ == "__main__":
    main()
