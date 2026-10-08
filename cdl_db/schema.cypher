CREATE NODE TABLE `Tag` (`keyword` STRING, PRIMARY KEY(`keyword`));
CREATE NODE TABLE `Talk` (`talk_id` STRING,`title` STRING,`category` STRING,`url` STRING,`description` STRING,`type` STRING,`video` STRING,`heysummit` STRING,`transcript` STRING,`description_source` STRING, PRIMARY KEY(`talk_id`));
CREATE NODE TABLE `Speaker` (`name` STRING, PRIMARY KEY(`name`));
CREATE NODE TABLE `Event` (`name` STRING,`description` STRING,`url` STRING, PRIMARY KEY(`name`));
CREATE NODE TABLE `Category` (`name` STRING, PRIMARY KEY(`name`));
CREATE REL TABLE `IS_DESCRIBED_BY` (FROM `Talk` TO `Tag`, `source` STRING,MANY_MANY);
CREATE REL TABLE `GIVES_TALK` (FROM `Speaker` TO `Talk`, `date` DATE,MANY_MANY);
CREATE REL TABLE `IS_PART_OF` (FROM `Talk` TO `Event`, MANY_MANY);
CREATE REL TABLE `IS_CATEGORIZED_AS` (FROM `Talk` TO `Category`, MANY_MANY);
