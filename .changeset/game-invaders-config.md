---
'xxscreeps': minor
---

Add a `game.invaders` configuration flag. Invasion waves are scheduled by a
room's cumulative harvest, so a scenario that models economy without defense
loses structures to an adversary it cannot answer. Setting `game.invaders: false`
stops new waves; invader cores, the invader NPC, and existing invader creeps are
untouched.
