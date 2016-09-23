<p align="center">
<b>-- SuperFish -- DataHack2016 -- Jerusalem --</b>
</p>

## Taxi Data Challenge
Problem Definition: Given 2,000,000 Taxi Rides in Manhattan Area, predict travel time for a ride.
2-Steps Model: ~100 GradientBoostRegressors for each trip between clusters, ~8 GradientBoostRegressors for each time-block of day.

## Requirements

- pandas
- shapely
- matplotlib
- mplleaflet

```
                                 .
                                A       ;
                      |   ,--,-/ \---,-/|  ,
                     _|\,'. /|      /|   `/|-.
                 \`.'    /|      ,            `;.
                ,'\   A     A         A   A _ /| `.;
              ,/  _              A       _  / _   /|  ;
             /\  / \   ,  ,           A  /    /     `/|
            /_| | _ \         ,     ,             ,/  \
           // | |/ `.\  ,-      ,       ,   ,/ ,/      \/
           / @| |@  / /'   \  \      ,              >  /|    ,--.
          |\_/   \_/ /      |  |           ,  ,/        \  ./' __:..
          |  __ __  |       |  | .--.  ,         >  >   |-'   /     `
        ,/| /  '  \ |       |  |     \      ,           |    /
       /  |<--.__,->|       |  | .    `.        >  >    /   (
      /_,' \\  ^  /  \     /  /   `.    >--            /^\   |
            \\___/    \   /  /      \__'     \   \   \/   \  |
             `.   |/          ,  ,                  /`\    \  )
               \  '  |/    ,       V    \          /        `-\
                `|/  '  V      V           \    \.'            \_
                 '`-.       V       V        \./'\
                     `|/-.      \ /   \ /,---`\
                      /   `._____V_____V'
                                 '     '
```
