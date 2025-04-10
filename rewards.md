1. **Vision-Based Rewards:**
   - `VISION_PROXIMITY_MAX_BONUS = 200` (Base vision reward/penalty)
   - `CATCH_BONUS = 1000` (For seekers catching hiders)
   - `SEEN_PENALTY_MULTIPLIER = 1.5` (Multiplier for hider penalty when seen)
   - `WALL_PENALTY = 10` (For hitting walls/closed doors)

2. **Exploration Rewards:**
   - +60 for distance > 2000 from start
   - +50 for distance > 1000
   - +25 for distance > 500
   - -5 for staying too close (<10 cells)
   - -150 for distance < 500

3. **Region/Door Rewards (Hiders):**
   - Base region reward: +300
   - Region B: +600 total (+300 base + 300 bonus)
     - Closing door: +500
     - Opening door: -500
   - Region C: +300
     - Closing door: +300
     - Opening door: -300
   - Region A:
     - Opening door: +200

4. **Seeker Door Rewards:**
   - Opening non-wall doors: +500

5. **Same Room Interactions:**
   - Hider-Hider in safe rooms: +500
   - Hider-Seeker:
     - <200 distance: -100
     - <500 distance: -50
   - Seeker-Hider:
     - <200 distance: +200
     - <500 distance: +100

6. **Rank Points (Hiders):**
   - Region B: +3
   - Region C: +2
   - Region A: +1
   - Other regions: +1

All these rewards are combined in the `step()` method to calculate the total reward for each action.