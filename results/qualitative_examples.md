# Qualitative examples (LSTM + attention, greedy)

## New regexes (test)

### Correct

- show me rows which never have an alphabetic character right before the substring 'Hello' and contain a vowel followed later by 's'
  - reference: `(~(.*[A-Za-z]__w0__.*))&(.*[AEIOUaeiou].*__w1__.*)`
  - generated: `(~(.*[A-Za-z]__w0__.*))&(.*[AEIOUaeiou].*__w1__.*)`
- a regular expression for text lines that have exactly 20 characters and are longer than 3 characters
  - reference: `(.{20})&(~(.{0,3}))`
  - generated: `(.{20})&(~(.{0,3}))`
- lines with at least 11 characters, with a vowel somewhere after a capital, and containing a vowel
  - reference: `(.{11,})&(.*[A-Z].*[AEIOUaeiou].*)&(.*[AEIOUaeiou].*)`
  - generated: `(.{11,})&(.*[A-Z].*[AEIOUaeiou].*)&(.*[AEIOUaeiou].*)`
- a regex for lines in which the text doesn't begin with a digit and the text finishes with the substring 'report'
  - reference: `(~([0-9].*))&(.*__w0__)`
  - generated: `(~([0-9].*))&(.*__w0__)`
- find a string that does not begin with a number and also terminates with the string 'water'
  - reference: `(~([0-9].*))&(.*__w0__)`
  - generated: `(~([0-9].*))&(.*__w0__)`

### Written differently, same behaviour

- a line containing three consecutive copies of 'light' and also lacking a number
  - reference: `(.*(__w0__){3,}.*)&(~(.*[0-9].*))`
  - generated: `(.*(__w0__){3,}.*)&(~(.*[0-9].*[0-9].*))`

### Wrong

- match text lines that contain a vowel less than three times, begin with a numeric character, and do not contain an uppercase letter before a vowel
  - reference: `(~((.*[AEIOUaeiou]){3,}.*))&([0-9].*)&(~(.*[A-Z].*[AEIOUaeiou].*))`
  - generated: `(~((.*[AEIOUaeiou]){3,}.*))&([0-9].*)&(~(.*[A-Z].*[AEIOUaeio`
- I want a line with a lowercase letter and lacking a number
  - reference: `(.*[a-z].*)&(~(.*[0-9].*))`
  - generated: `(.*[a-z].*)&(~(.*[0-9].*[0-9].*))`
- get lines with a letter of the alphabet somewhere after a vowel, with a numeral immediately followed by 'house', and with a vowel immediately followed by a lower-case letter
  - reference: `(.*[AEIOUaeiou].*[A-Za-z].*)&(.*[0-9]__w0__.*)&(.*[AEIOUaeiou][a-z].*)`
  - generated: `(.*[AEIOUaeiou].*[A-Za-z].*)&(.*[0-9]__w0__.*)&(.*[AEIOUaeiou][a-`
- find all lines that do not contain a space but do not contain the sequence 'Alice' immediately followed by a vowel
  - reference: `(~(.* .*))&(~(.*__w0__[AEIOUaeiou].*))`
  - generated: `(~(.* [AEIOUaeiou].*))&(~(.*__w0__[AEIOUaeiou].*))`
- I need strings ending in the sequence 'purple' or the substring 'offer'
  - reference: `(.*__w0__)|(.*__w1__)`
  - generated: `(.*__w0__)|(.*__w1__.*)`
- I need any text that ends with the string '99' or '42'
  - reference: `(.*__w0__)|(.*__w1__)`
  - generated: `(.*__w0__)|(.*__w1__.*)`

## New phrasings

### Correct

- I want all strings where 'table' is repeated 4 times in a row and the length is at least 9
  - reference: `(.*(__w0__){4,}.*)&(.{9,})`
  - generated: `(.*(__w0__){4,}.*)&(.{9,})`
- show me all text lines that contain a lower-case letter and later a capital letter or have the substring 'bridge' right before a letter of the alphabet
  - reference: `(.*[a-z].*[A-Z].*)|(.*__w0__[A-Za-z].*)`
  - generated: `(.*[a-z].*[A-Z].*)|(.*__w0__[A-Za-z].*)`
- get text lines that are made up of only lowercase letters and have fewer than 5 characters
  - reference: `([a-z]+)&(~(.{5,}))`
  - generated: `([a-z]+)&(~(.{5,}))`
- all strings which contain a vowel immediately after the text 'un' and contain a letter of the alphabet 6 or more times
  - reference: `(.*__w0__[AEIOUaeiou].*)&((.*[A-Za-z]){6,}.*)`
  - generated: `(.*__w0__[AEIOUaeiou].*)&((.*[A-Za-z]){6,}.*)`
- get each line where the line does not start with the substring 'window' and the line ends with a lowercase letter
  - reference: `(~(__w0__.*))&(.*[a-z])`
  - generated: `(~(__w0__.*))&(.*[a-z])`

### Written differently, same behaviour

- select text lines which are no longer than 1 character and have a digit at the end
  - reference: `(.{0,1})&(.*[0-9])`
  - generated: `(.{0,1})&(.*[0-9].*)`
- give me every line that has the text "q" at the start, contains a lowercase letter with a capital right after, and has 'q' at least three times
  - reference: `(__w0__.*)&(.*[a-z][A-Z].*)&((.*__w0__){3,}.*)`
  - generated: `(__w0__.*)&(.*[A-Z][A-Z].*)&((.*__w0__){3,}.*)`
- I want all lines that have the sequence '5' fewer than 4 times, consist only of lowercase letters, and have an alphabetic character at the end
  - reference: `(~((.*__w0__){4,}.*))&([a-z]+)&(.*[A-Za-z])`
  - generated: `(~((.*__w0__){4,}.*))&([a-z]+)&(.*[A-Za-z].*)`

### Wrong

- select any string that contains the substring "x" no fewer than four times or is two or more characters long
  - reference: `((.*__w0__){4,}.*)|(.{2,})`
  - generated: `(~((.*__w0__){4,}.*))|(.{2,})`
- a regular expression for strings which include the string 'rome' six times or more, contain the sequence 'rome' no fewer than 6 times and have more than seven characters
  - reference: `((.*__w0__){6,}.*)&((.*__w0__){6,}.*)&(~(.{0,7}))`
  - generated: `((.*__w0__){6,}.*)&(~((.*__w0__){6,}.*))&(~(.{0,7}))`
- lines that contain a letter and later a numeric character or have a letter at the end
  - reference: `(.*[A-Za-z].*[0-9].*)|(.*[A-Za-z])`
  - generated: `(.*[A-Za-z].*[0-9].*)|(.*[A-Za-z].*)`
- all strings that have the substring 'chair' at the end or contain an upper-case letter directly followed by the substring 'blue'
  - reference: `(.*__w0__)|(.*[A-Z]__w1__.*)`
  - generated: `(__w0__.*)|(.*[A-Z]__w1__.*)`
- show me all strings that have a numeral at the end and have a numeral at the end
  - reference: `(.*[0-9])&(.*[0-9])`
  - generated: `(.*[0-9].*)&(.*[0-9].*)`
- regex matching all lines that have the sequence "Alice" at the end but do not contain a space
  - reference: `(.*__w0__)&(~(.* .*))`
  - generated: `(__w0__.*)&(~(.* .*))`

## Handwritten

### Correct

- lines with the word 'error' in them
  - reference: `.*__w0__.*`
  - generated: `.*__w0__.*`
- any line that mentions 'invoice'
  - reference: `.*__w0__.*`
  - generated: `.*__w0__.*`
- text that begins with 'Dear'
  - reference: `__w0__.*`
  - generated: `__w0__.*`
- lines finishing with a digit
  - reference: `.*[0-9]`
  - generated: `.*[0-9]`
- lines whose first character is a capital letter
  - reference: `[A-Z].*`
  - generated: `[A-Z].*`

### Wrong

- lines with no digits at all
  - reference: `~(.*[0-9].*)`
  - generated: `.{0,9}`
- lines of at most 10 characters that contain a digit
  - reference: `(.{0,10})&(.*[0-9].*)`
  - generated: `(.{0,10})|(.*[0-9].*)`
