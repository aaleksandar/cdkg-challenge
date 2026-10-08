COPY `Tag` (`keyword`) FROM "Tag.csv" (header=true, parallel=true);
COPY `Talk` (`talk_id`,`title`,`category`,`url`,`description`,`type`,`video`,`heysummit`,`transcript`,`description_source`) FROM "Talk.csv" (header=true, parallel=false);
COPY `Speaker` (`name`) FROM "Speaker.csv" (header=true, parallel=true);
COPY `Event` (`name`,`description`,`url`) FROM "Event.csv" (header=true, parallel=true);
COPY `Category` (`name`) FROM "Category.csv" (header=true, parallel=true);
COPY `IS_DESCRIBED_BY` (`source`) FROM "IS_DESCRIBED_BY_Talk_Tag.csv" (to='Tag', from='Talk', header=true, parallel=true);
COPY `GIVES_TALK` (`date`) FROM "GIVES_TALK_Speaker_Talk.csv" (to='Talk', from='Speaker', header=true, parallel=true);
COPY `IS_PART_OF` FROM "IS_PART_OF_Talk_Event.csv" (to='Event', from='Talk', header=true, parallel=true);
COPY `IS_CATEGORIZED_AS` FROM "IS_CATEGORIZED_AS_Talk_Category.csv" (to='Category', from='Talk', header=true, parallel=true);
