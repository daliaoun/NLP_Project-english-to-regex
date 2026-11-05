# Qualitative examples (lstm_attn_aug_s1, greedy, test set)

## Correct (exact match)

- **Input:** Items terminating with 'dog' preceding a vowel.
  - reference: `(.*)(dog.*[AEIOUaeiou].*)`
  - generated: `(.*)(dog.*[AEIOUaeiou].*)`
- **Input:** Items with a numeral or an upper case letter at least five times.
  - reference: `(([A-Z])|([0-9])){5,}`
  - generated: `(([A-Z])|([0-9])){5,}`
- **Input:** lines with 'dog' before a capital letter followed by a lower-case letter
  - reference: `(dog).*([A-Z].*[a-z].*).*`
  - generated: `(dog).*([A-Z].*[a-z].*).*`
- **Input:** lines not words with lower-case letter, 7 or more times
  - reference: `~(\b([a-z]){7,}\b)`
  - generated: `~(\b([a-z]){7,}\b)`
- **Input:** lines not having lower-case letter or containing a capital
  - reference: `(~([a-z]))|(.*[A-Z].*)`
  - generated: `(~([a-z]))|(.*[A-Z].*)`
- **Input:** lines with dog, lower-case letters, or vowels before a letter
  - reference: `((dog)|([a-z])|([AEIOUaeiou])).*([A-Za-z]).*`
  - generated: `((dog)|([a-z])|([AEIOUaeiou])).*([A-Za-z]).*`

## Different string, same behaviour

- **Input:** lines with a character before 'dog'
  - reference: `((.)+).*(dog).*`
  - generated: `(.*..*).*(dog).*`
- **Input:** Items with 'dog' preceding a small letter or letter.
  - reference: `(dog).*(([A-Za-z])|([a-z])).*`
  - generated: `(dog).*(([a-z])|([A-Za-z])).*`
- **Input:** Lines containing 2 or more characters or numbers.
  - reference: `(.*(.)|([0-9]).*){2,}`
  - generated: `((.)|([0-9])){2,}`

## Failures

- **Input:** lines containing a character followed by a number, 6 or more times
  - reference: `(.).*(([0-9]){6,}).*`
  - generated: `(.*..*[0-9].*.*){6,}`
- **Input:** lines ending in zero or more of a letter and string 'dog'
  - reference: `(.*)((([A-Za-z])*)|(dog))`
  - generated: `((.*)([A-Za-z]))&((dog)*)`
- **Input:** no lines containing the string 'dog' or a capital letter
  - reference: `.*(~(dog))&([A-Z]).*`
  - generated: `.*(dog)|([A-Z]).*`
- **Input:** lines that have a character which is numbering at 4 or greater appearances and that precedes a small letter
  - reference: `((.){4,}).*([a-z]).*`
  - generated: `((.){4,})&([a-z].*..*)`
- **Input:** Items with a numeral or a character.
  - reference: `((.)|([0-9]))*`
  - generated: `.*(.)|([0-9]).*`
- **Input:** lines with words with string 'dog' and vowel or a capital
  - reference: `\b(dog)&(([AEIOUaeiou])|([A-Z]))\b`
  - generated: `\b((dog)&([AEIOUaeiou]))|([A-Z])\b`
- **Input:** lines which do not contain the string "dog" or a vowel or a letter
  - reference: `~((dog)&([AEIOUaeiou])&([A-Za-z]))`
  - generated: `~(((dog)|([AEIOUaeiou]))|([A-Za-z]))`
- **Input:** all lines
  - reference: `((.*)((dog)|(truck)))*`
  - generated: `((.*)([A-Z]))*`
