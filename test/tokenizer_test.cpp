#include <gtest/gtest.h>
#include "tokenizer/greedy_tokenizer.hpp"

// TODO: make these tests modular for swapping cl100k and o200k
TEST(TokenizerTest, StringOnly) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("Hello world");
    std::vector<int> expected {9906, 1917};
    EXPECT_EQ(expected, actual);
}


TEST(TokenizerTest, NumbersOnly) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("123093934820905827983");
    std::vector<int> expected {4513, 25202, 24347, 18248, 22393, 24920, 24742};
    EXPECT_EQ(expected, actual);
}

TEST(TokenizerTest, StringAndNumbers) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("In the 2020 United States census, Trenton was the fifth largest city in Central Jersey and the 10th-most-populous municipality statewide,[28] with a population of 90,871,[9][10] an increase of 5,958 (+7.0%) from the 2010 census count of 84,913,[29][30] which in turn had reflected a decline of 490 (−0.6%) from the 85,403 counted in the 2000 census.[31] The Census Bureau's Population Estimates Program calculated that the city's population was 89,661 in 2022,[9] ranking the city the 382nd-most-populous in the country.[11]");
    std::vector<int> expected {644, 279, 220, 2366, 15, 3723, 4273, 44702, 11, 
        48620, 263, 574, 279, 18172, 7928, 3363, 304, 10913, 16228, 323, 279, 
        220, 605, 339, 63498, 41352, 13850, 57918, 52041, 17706, 1591, 60, 449,
        264, 7187, 315, 220, 1954, 11, 25665, 17706, 24, 1483, 605, 60, 459, 
        5376, 315, 220, 20, 11, 27079, 18457, 22, 13, 15, 11587, 505, 279, 
        220, 679, 15, 44702, 1797, 315, 220, 5833, 11, 24331, 17706, 1682, 1483, 
        966, 60, 902, 304, 2543, 1047, 27000, 264, 18174, 315, 220, 18518, 320, 
        34363, 15, 13, 21, 11587, 505, 279, 220, 5313, 11, 13074, 31094, 304, 279,
        220, 1049, 15, 44702, 8032, 2148, 60, 578, 46627, 22555, 596, 40629, 95619, 
        6826, 16997, 430, 279, 3363, 596, 7187, 574, 220, 4578, 11, 24132, 304, 
        220, 2366, 17, 17706, 24, 60, 23862, 279, 3363, 279, 220, 18781, 303, 
        63498, 41352, 13850, 304, 279, 3224, 8032, 806, 60, 220};
    EXPECT_EQ(expected, actual);
}


TEST(TokenizerTest, StringAndNumbersAndSpecialChars) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize(R"(The year is 2024! Email me@test.com or call +1-555-123-4567.
Price: $49.99 (50% off!) https://example.com/path?id=42&token=abc123
Code: int x = 10; // comment
Math: 2+2=4, π≈3.14159, √16=4
Special: !@#$%^&*()_+-=[]{}|;':",.<>?/`~
Unicode: 你好 🚀 café naïve
Mixed: abc123XYZ, test_var2, CamelCase99, snake_case_123
Quotes: "double" 'single' `backtick`
Escapes: \n\t\r\\ \"
Empty:

Trailing spaces:   
Tab	separated	values
Multiple!!!Punctuation???Marks...
hyphenated-words and under_scored_text
ALLCAPS lowercase MiXeDcAsE
Numbers: 0, -42, 3.14, 1e10, 0xFF, 0b1010
End.)");
    std::vector<int> expected {791, 1060, 374, 220, 2366, 19, 0, 8463, 757, 48427, 
        916, 477, 1650, 489, 16, 12, 14148, 12, 4513, 12, 10961, 22, 627, 7117, 25, 
        400, 2491, 13, 1484, 320, 1135, 4, 1022, 16715, 3788, 1129, 8858, 916, 52076, 
        20970, 28, 2983, 5, 5963, 28, 13997, 4513, 198, 2123, 25, 528, 865, 284, 220, 
        605, 26, 443, 4068, 198, 8991, 25, 220, 17, 10, 17, 28, 19, 11, 52845, 60094, 
        230, 18, 13, 9335, 2946, 11, 12264, 248, 845, 28, 19, 198, 20989, 25, 758, 31, 
        49177, 46999, 5, 9, 368, 62, 22192, 31792, 6390, 91, 26, 1232, 498, 16134, 48685, 
        14, 63, 89241, 35020, 25, 220, 57668, 53901, 11410, 248, 222, 53050, 95980, 
        588, 198, 87533, 25, 40122, 4513, 33296, 11, 1296, 4715, 17, 11, 69254, 4301, 
        1484, 11, 26332, 19640, 62, 4513, 198, 44880, 25, 330, 4429, 1, 364, 15698, 6, 
        1595, 1445, 35255, 4077, 37221, 9521, 25, 1144, 77, 5061, 12285, 3505, 1144, 
        702, 3606, 1473, 1305, 14612, 12908, 25, 5996, 8750, 85786, 50700, 47039, 198, 
        33189, 12340, 47, 73399, 34115, 90470, 9522, 8671, 15112, 660, 12, 5880, 323, 
        1234, 10622, 67, 4424, 198, 4006, 32500, 50, 43147, 21402, 55, 68, 35, 66, 2170, 
        36, 198, 28336, 25, 220, 15, 11, 482, 2983, 11, 220, 18, 13, 975, 11, 220, 16, 
        68, 605, 11, 220, 15, 9448, 11, 220, 15, 65, 4645, 15, 198, 3812, 13};
    EXPECT_EQ(expected, actual);
}

TEST(TokenizerTest, LongTest) {
    GreedyTokenizer tok;
std::vector<int> actual = tok.tokenize(R"(Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?
I see thee yet, in form as palpable
As this which now I draw.
Thou marshall'st me the way that I was going;
And such an instrument I was to use.
Mine eyes are made the fools o' the other senses,
Or else worth all the rest; I see thee still,
And on thy blade and dudgeon gouts of blood,
Which was not so before. There's no such thing:
It is the bloody business which informs
Thus to mine eyes. Now o'er the one halfworld
Nature seems dead, and wicked dreams abuse
The curtain'd sleep; witchcraft celebrates
Pale Hecate's offerings, and wither'd murder,
Alarum'd by his sentinel, the wolf,
Whose howl's his watch, thus with his stealthy pace.
With Tarquin's ravishing strides, towards his design
Moves like a ghost. Thou sure and firm-set earth,
Hear not my steps, which way they walk, for fear
Thy very stones prate of my whereabout,
And take the present horror from the time,
Which now suits with it. Whiles I threat, he lives:
Words to the heat of deeds too cold breath gives.
[A bell rings]
I go, and it is done; the bell invites me.
Hear it not, Duncan; for it is a knell
That summons thee to heaven or to hell.)");

    std::vector<int> expected {3957, 420, 264, 40331, 902, 358, 1518, 1603, 757, 345, 
        791, 3790, 9017, 856, 1450, 30, 15936, 11, 1095, 757, 43789, 40344, 627, 40, 
        617, 40344, 539, 11, 323, 3686, 358, 1518, 40344, 2103, 627, 9470, 34223, 539, 
        11, 19094, 11376, 11, 38761, 198, 1271, 8430, 439, 311, 14254, 30, 477, 1989, 
        34223, 719, 198, 32, 40331, 315, 279, 4059, 11, 264, 905, 9886, 345, 85438, 287,
        505, 279, 8798, 30592, 14655, 8271, 5380, 40, 1518, 40344, 3686, 11, 304, 1376, 
        439, 75649, 481, 198, 2170, 420, 902, 1457, 358, 4128, 627, 1016, 283, 3678, 19549, 
        596, 83, 757, 279, 1648, 430, 358, 574, 2133, 280, 3112, 1778, 459, 14473, 358, 574, 
        311, 1005, 627, 64595, 6548, 527, 1903, 279, 84742, 297, 6, 279, 1023, 38207, 345, 
        2244, 775, 5922, 682, 279, 2800, 26, 358, 1518, 40344, 2103, 345, 3112, 389, 26236, 
        25879, 323, 294, 20132, 263, 342, 11934, 315, 6680, 345, 23956, 574, 539, 779, 1603, 
        13, 2684, 596, 912, 1778, 3245, 512, 2181, 374, 279, 36277, 2626, 902, 64252, 198, 
        45600, 311, 10705, 6548, 13, 4800, 297, 94678, 279, 832, 4376, 14957, 198, 79519, 
        5084, 5710, 11, 323, 45077, 19226, 11737, 198, 791, 46866, 4265, 6212, 26, 37482, 
        7868, 49193, 198, 78666, 473, 762, 349, 596, 33935, 11, 323, 449, 261, 4265, 10102, 
        345, 2149, 277, 372, 4265, 555, 813, 81878, 11, 279, 37642, 345, 1671, 974, 1268, 75, 
        596, 813, 3821, 11, 8617, 449, 813, 48065, 88, 18338, 627, 2409, 24912, 36444, 596, 
        43643, 11218, 37387, 11, 7119, 813, 2955, 198, 46889, 1093, 264, 20457, 13, 86471, 
        2771, 323, 7626, 25063, 9578, 345, 39, 686, 539, 856, 7504, 11, 902, 1648, 814, 4321, 
        11, 369, 8850, 198, 1016, 88, 1633, 27302, 550, 349, 315, 856, 1405, 9274, 345, 3112, 
        1935, 279, 3118, 22169, 505, 279, 892, 345, 23956, 1457, 29076, 449, 433, 13, 1254, 
        3742, 358, 6023, 11, 568, 6439, 512, 24390, 311, 279, 8798, 315, 54811, 2288, 9439, 
        11745, 6835, 627, 23335, 29519, 25562, 933, 40, 733, 11, 323, 433, 374, 2884, 26, 279, 
        29519, 45510, 757, 627, 39, 686, 433, 539, 11, 42409, 26, 369, 433, 374, 264, 1168, 
        616, 198, 4897, 75400, 40344, 311, 23070, 477, 311, 15123, 13
    };

    EXPECT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        std::cout << i << " is " << (expected[i] == actual[i] ? "OK, " : " WRONG, ") << expected[i] << " and got " << actual[i] << "\n";
    }
    EXPECT_EQ(expected, actual);
};
