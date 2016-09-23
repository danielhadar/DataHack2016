<p align="center">
<b>-- SuperFish -- DataHack2016 -- Jerusalem --</b>
</p>

## Taxi Data Challenge
Problem Definition: Given 2,000,000 Taxi Rides in Manhattan Area, predict travel time for a ride.
2-Steps Model: ~100 GradientBoostRegressors for each trip between clusters, ~8 GradientBoostRegressors for each time-block of day.
<br>
To run see 'main.py'.
<br>
Data could be found <a href="https://www.dropbox.com/sh/ucx5z0ck5wh4so2/AABBuNoFafDtZ4tuYoZ4qoLOa?dl=0">Here</a>.

## Requirements

- pandas
- shapely
- matplotlib
- mplleaflet (visualisations only)

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
